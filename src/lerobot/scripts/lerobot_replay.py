# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of an episode from a dataset on a robot.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Examples:

```shell
lerobot-replay \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=black \
    --dataset.repo_id=<USER>/record-test \
    --dataset.episode=0
```

Example replay with bimanual so100:
```shell
lerobot-replay \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --dataset.repo_id=${HF_USER}/bimanual-so100-handover-cube \
  --dataset.episode=0
```

"""

import json
import logging
import math
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from threading import Event
from typing import Any

import numpy as np
from PIL import Image

from lerobot.cameras import (  # noqa: F401
    CameraConfig,
    ColorMode,
    make_cameras_from_configs,
    opencv,
    realsense,
)
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.lerobot_types import RobotAction
from lerobot.processor import (
    make_default_robot_action_processor,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_piperx_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    bi_yam_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    sim_robot,
    so_follower,
    unitree_g1,
    yam_follower,
)
from lerobot.robots.utils import check_control_rate, episode_ended
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.keyboard_input import create_key_listener
from lerobot.utils.rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path'). If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the policy fps.
    fps: int = 30


@dataclass
class ReplayConfig:
    robot: RobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Write each target and its measured feedback to a new JSONL file.
    trace_path: Path | None = None
    # A standalone RGB camera enables a preview before connecting the robot.
    layout_camera: CameraConfig | None = None
    # Defaults to the first frame from layout_dataset_camera in the selected episode.
    layout_reference_path: Path | None = None
    layout_dataset_camera: str = "fixed_front"
    layout_alpha: float = 0.5
    layout_display_ip: str | None = None
    layout_display_port: int | None = None


def _confirm_initial_layout(cfg: ReplayConfig, dataset: LeRobotDataset, camera_config: CameraConfig) -> bool:
    if not 0 <= cfg.layout_alpha <= 1:
        raise ValueError("layout_alpha must be in [0, 1].")

    if cfg.layout_reference_path is not None:
        with Image.open(cfg.layout_reference_path) as image:
            reference = image.convert("RGB")
    else:
        key = f"{OBS_IMAGES}.{cfg.layout_dataset_camera}"
        if not dataset.num_frames or key not in dataset.features:
            raise ValueError(
                f"Episode {cfg.dataset.episode} has no {key} reference. Set layout_reference_path."
            )
        pixels = np.asarray(dataset[0][key])
        if pixels.ndim == 3 and pixels.shape[0] == 3 and pixels.shape[-1] != 3:
            pixels = pixels.transpose(1, 2, 0)
        if pixels.ndim != 3 or pixels.shape[-1] != 3:
            raise ValueError(f"Expected an RGB reference at {key}, got shape {pixels.shape}.")
        if np.issubdtype(pixels.dtype, np.floating):
            pixels = np.rint(np.clip(pixels, 0, 1) * 255)
        reference = Image.fromarray(pixels.astype(np.uint8))

    displayed = Event()
    confirmed = Event()
    cancelled = Event()

    def on_key(key: str) -> None:
        if key.lower() in ("esc", "q"):
            cancelled.set()
        elif key.lower() in ("right", "n") and displayed.is_set():
            confirmed.set()

    camera = make_cameras_from_configs({"layout": camera_config})["layout"]
    with ExitStack() as resources:
        listener = create_key_listener(on_key, controls_help="Right/n to start replay, Esc/q to cancel")
        if listener is None:
            raise RuntimeError("Initial layout requires a keyboard listener or an interactive terminal.")
        resources.callback(listener.stop)
        resources.enter_context(camera)
        init_rerun("lerobot_replay_layout", ip=cfg.layout_display_ip, port=cfg.layout_display_port)
        resources.callback(shutdown_rerun)
        logging.info("Arrange episode %d. Right/n starts replay, Esc/q cancels.", cfg.dataset.episode)
        overlay_reference = reference
        while not (confirmed.is_set() or cancelled.is_set()):
            frame = camera.async_read()
            if getattr(camera_config, "color_mode", ColorMode.RGB) == ColorMode.BGR:
                frame = frame[..., ::-1]
            live = Image.fromarray(frame).convert("RGB")
            if overlay_reference.size != live.size:
                overlay_reference = reference.resize(live.size, Image.Resampling.BILINEAR)
            overlay = Image.blend(live, overlay_reference, cfg.layout_alpha)
            log_rerun_data(observation={"initial_layout": np.asarray(overlay)}, compress_images=True)
            displayed.set()
            cancelled.wait(0.1)
    return confirmed.is_set() and not cancelled.is_set()


def _trace_positions(values: RobotAction, names: list[str]) -> dict[str, float | None]:
    positions = {name: float(values[name]) for name in names if name in values}
    return {name: value if math.isfinite(value) else None for name, value in positions.items()}


@parser.wrap()
def replay(cfg: ReplayConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot_action_processor = make_default_robot_action_processor()

    robot = make_robot_from_config(cfg.robot)
    dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[cfg.dataset.episode])

    actions = dataset.select_columns(ACTION)
    check_control_rate(robot, dataset.fps)

    # Replay must hit the dataset's own frame rate, or the trajectory plays back at the
    # wrong speed. The optional trace does not change the playback rate.
    timer = CycleTimer(dataset.fps, records_data=False)
    action_names = dataset.features[ACTION]["names"]
    piper = robot if isinstance(robot, bi_piperx_follower.BiPiperXFollower) else None

    with ExitStack() as resources:
        if cfg.layout_camera is not None and not _confirm_initial_layout(cfg, dataset, cfg.layout_camera):
            logging.info("Replay cancelled during initial layout.")
            return
        trace = None
        if cfg.trace_path is not None:
            trace = resources.enter_context(cfg.trace_path.open("x", encoding="utf-8", buffering=1))
            logging.info("Writing replay trace to %s", cfg.trace_path)
        robot.connect()
        resources.callback(robot.disconnect)
        resources.callback(timer.log_run_summary)
        log_say("Replaying episode", cfg.play_sounds, blocking=True)
        started_at = time.perf_counter()
        pending: dict[str, Any] | None = None
        try:
            for idx in range(dataset.num_frames):
                timer.tick()

                with timer.section("read_frame"):
                    action_array = actions[idx][ACTION]
                    action = {name: action_array[i] for i, name in enumerate(action_names)}

                with timer.section("observe"):
                    robot_obs = robot.get_observation()

                if trace is not None:
                    with timer.section("trace"):
                        sensed_at_s = time.perf_counter() - started_at
                        sensed = _trace_positions(robot_obs, action_names)
                        feedback = piper.get_replay_feedback() if piper is not None else None
                        if pending is not None:
                            previous, pending = pending, None
                            previous["sensed_after"] = sensed
                            previous["sensed_after_s"] = sensed_at_s
                            if piper is not None:
                                previous["feedback_after"] = feedback
                            trace.write(json.dumps(previous, allow_nan=False) + "\n")
                        pending = {
                            "episode": cfg.dataset.episode,
                            "frame": idx,
                            "target": _trace_positions(action, action_names),
                            "sensed_before": sensed,
                            "sensed_before_s": sensed_at_s,
                            "send_completed_s": None,
                            "sent_target": None,
                            "sensed_after": None,
                            "sensed_after_s": None,
                        }
                        if piper is not None:
                            pending["joint_command"] = piper.config.joint_command
                            pending["feedback_before"] = feedback
                            pending["feedback_after"] = None
                            pending["send_completed_unix_s"] = None

                with timer.section("send"):
                    processed_action = robot_action_processor((action, robot_obs))
                    sent_action = robot.send_action(processed_action)
                    if pending is not None:
                        pending["send_completed_s"] = time.perf_counter() - started_at
                        if piper is not None:
                            pending["send_completed_unix_s"] = time.time()
                        pending["sent_target"] = _trace_positions(sent_action, action_names)

                if episode_ended(robot):
                    logging.info(
                        "Sim episode ended at frame %d of %d, stopping the replay",
                        idx + 1,
                        dataset.num_frames,
                    )
                    break

                timer.wait()

            if pending is not None:
                # The final command has no next control tick to supply its feedback.
                final_obs = robot.get_observation()
                sensed_at_s = time.perf_counter() - started_at
                pending["sensed_after"] = _trace_positions(final_obs, action_names)
                pending["sensed_after_s"] = sensed_at_s
                if piper is not None:
                    pending["feedback_after"] = piper.get_replay_feedback()
        finally:
            if trace is not None and pending is not None:
                trace.write(json.dumps(pending, allow_nan=False) + "\n")


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
