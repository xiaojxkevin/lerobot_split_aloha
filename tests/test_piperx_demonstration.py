"""Exercise the Split-ALOHA episode and dataset path without opening devices."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.datasets import LeRobotDataset
from lerobot.model import piperx_dynamics
from lerobot.piperx import (
    ACTION_KEYS,
    LEGACY_GRIPPER_PER_METER,
    PiperArmConfig,
    PiperXArm,
    PiperXFeedbackUnavailable,
)
from lerobot.robots.bi_piperx_follower import BiPiperXFollower, BiPiperXFollowerConfig
from lerobot.robots.bi_piperx_follower.bi_piperx_follower import PiperArm
from lerobot.scripts import lerobot_record, piperx_demonstration
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, TeleoperationDatasetConfig, teleoperate
from lerobot.teleoperators.bi_piperx_leader import BiPiperXLeader, BiPiperXLeaderConfig, mit_control
from lerobot.teleoperators.bi_piperx_leader.bi_piperx_leader import PiperXLeaderArm

pytestmark = pytest.mark.dataset


class FakeArm(PiperArm, PiperXLeaderArm):
    ARM_STATUS = SimpleNamespace(
        CtrlMode=SimpleNamespace(STANDBY=0, CAN_CTRL=1),
        ArmStatus=SimpleNamespace(NORMAL=0),
        ModeFeedback=SimpleNamespace(MOVE_J=1, MOVE_MIT=6),
    )
    OPTIONS = SimpleNamespace(MOTION_MODE=SimpleNamespace(MIT=6))

    def __init__(self, name: str) -> None:
        super().__init__(PiperArmConfig(port=name), max_feedback_age=0.25)
        self.driver = self
        self.gripper = self
        self.limits = [(-0.1, 0.1)] * 6
        self.mode = 1
        self.positions = [0.08] * 6
        self.gripper_targets: list[float] = []
        self.js_targets: list[list[float]] = []
        self.mit_ticks: list[float] = []
        self.gripper_enabled = True
        self.gripper_width_m = 0.0
        self.disable_gripper_calls = 0

    def get_status(self) -> SimpleNamespace:
        return SimpleNamespace(ctrl_mode=1, arm_status=0, mode_feedback=self.mode)

    def get_gripper_status(self) -> SimpleNamespace:
        return SimpleNamespace(
            msg=SimpleNamespace(
                mode="width",
                value=self.gripper_width_m,
                foc_status=SimpleNamespace(driver_enable_status=self.gripper_enabled),
            ),
            timestamp=time.time(),
        )

    def disable_gripper(self) -> bool:
        self.gripper_enabled = False
        self.disable_gripper_calls += 1
        return True

    def get_joint_enable_states(self) -> list[bool]:
        return [True] * 6

    def enable(self, timeout_s: float) -> None:
        pass

    def get_joint_positions(self) -> list[float]:
        return self.positions.copy()

    def set_auto_set_motion_mode_enabled(self, enabled: bool) -> None:
        pass

    def set_motion_mode(self, mode: int) -> None:
        self.mode = mode

    def move_mit(self, joint: int, *, p_des: float, kp: float, kd: float, t_ff: float) -> None:
        self.mit_ticks.append(kp)

    def move_js(self, positions: list[float]) -> None:
        self.mode = 1
        self.positions = positions
        self.js_targets.append(positions)

    def move_gripper_m(self, width: float) -> None:
        self.gripper_targets.append(width)

    def observe(self) -> list[float]:
        return [*self.get_joint_positions(), 0.0]


def test_teleoperate_records_two_mit_episodes_at_joint_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cameras = {
        name: RealSenseCameraConfig(serial_number_or_name=str(i), width=16, height=16, fps=30)
        for i, name in enumerate(("fixed_front", "left_arm", "right_arm"), start=1)
    }
    config = TeleoperateConfig(
        teleop=BiPiperXLeaderConfig(
            mit_zero_drag=True,
            mit_gripper_mode="closed",
            home_duration_s=0.02,
            calibration_dir=tmp_path / "calibration" / "leader",
        ),
        robot=BiPiperXFollowerConfig(cameras=cameras, calibration_dir=tmp_path / "calibration" / "follower"),
        dataset=TeleoperationDatasetConfig(
            repo_id="local/test",
            root=tmp_path / "data",
            single_task="Move the handles.",
            num_episodes=2,
            reset_time_s=0,
            encoder_threads=1,
        ),
    )
    assert config.dataset is not None
    config.dataset.rgb_encoder.vcodec = "libx264"
    config.dataset.rgb_encoder.preset = "ultrafast"
    # Use real action clipping and dataset writing with in-memory camera images.
    monkeypatch.setattr(
        "lerobot.robots.bi_piperx_follower.bi_piperx_follower.make_cameras_from_configs",
        lambda configs: {
            name: SimpleNamespace(
                height=16, width=16, read_latest=lambda: np.zeros((16, 16, 3), dtype=np.uint8)
            )
            for name in configs
        },
    )
    robot = BiPiperXFollower(config.robot)
    leader = BiPiperXLeader(config.teleop)
    leaders = tuple(FakeArm(name) for name in ("left_leader", "right_leader"))
    followers = tuple(FakeArm(name) for name in ("left_follower", "right_follower"))
    robot.arms, leader.arms = followers, leaders

    def connect_robot() -> None:
        robot._resources = ExitStack()
        robot._executor = robot._resources.enter_context(ThreadPoolExecutor(max_workers=2))

    monkeypatch.setattr(robot, "connect", connect_robot)
    monkeypatch.setattr(leader, "connect", lambda: setattr(leader, "_resources", ExitStack()))
    monkeypatch.setattr(lerobot_record, "make_robot_from_config", lambda _: robot)
    monkeypatch.setattr(lerobot_record, "make_teleoperator_from_config", lambda _: leader)
    monkeypatch.setattr(piperx_demonstration, "ensure_can_control", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        piperx_dynamics, "PiperXDynamics", lambda _: SimpleNamespace(torques=lambda *a, **kw: np.zeros(6))
    )
    monkeypatch.setattr(mit_control, "_JS_HANDOFF_HOLD_S", 0.01)
    monkeypatch.setattr(mit_control, "_MIT_MODE_CONFIRM_S", 0.01)
    monkeypatch.setattr(mit_control, "_MIT_RELEASE_S", 0.01)
    events = {"exit_early": False, "rerecord_episode": False, "stop_recording": False}
    monkeypatch.setattr(lerobot_record, "init_keyboard_listener", lambda: (None, events))

    def observe_during_episode() -> dict:
        assert all(arm.mode == 6 and 0 in arm.mit_ticks for arm in leaders)
        assert all(arm.mode == 1 and arm.positions == [0.0] * 6 for arm in followers)
        leaders[0].positions = [0.2] * 6
        leaders[1].positions = [-0.2] * 6
        events["exit_early"] = True
        return BiPiperXFollower.get_observation(robot)

    monkeypatch.setattr(robot, "get_observation", observe_during_episode)
    dataset = teleoperate(config)
    assert isinstance(dataset, LeRobotDataset)
    assert dataset.num_episodes == 2
    assert all(arm.mode == 1 and arm.js_targets[-1] == [0.0] * 6 for arm in (*leaders, *followers))
    assert all(not arm.mit_ticks and arm.gripper_targets == [0.0, 0.0] for arm in followers)
    assert all(not arm.gripper_targets for arm in leaders)
    assert all(arm.gripper_enabled for arm in leaders)

    info = json.loads((dataset.root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert set(dataset.meta.video_keys) == {f"observation.images.{name}" for name in cameras}
    assert len(list((dataset.root / "videos").rglob("*.mp4"))) == 3
    states = pq.read_table(dataset.root / "data", columns=["observation.state", "action"]).to_pydict()
    np.testing.assert_allclose(states["observation.state"], np.zeros((2, len(ACTION_KEYS))))
    expected_action = [0.1] * 6 + [0.0] + [-0.1] * 6 + [0.0]
    np.testing.assert_allclose(states["action"], [expected_action, expected_action])


def test_mit_send_failure_holds_measured_leader_poses(monkeypatch: pytest.MonkeyPatch) -> None:
    leaders = (FakeArm("left_leader"), FakeArm("right_leader"))
    followers = (FakeArm("left_follower"), FakeArm("right_follower"))
    robot = SimpleNamespace(arms=followers)
    leader = SimpleNamespace(arms=leaders, config=BiPiperXLeaderConfig(mit_zero_drag=True))
    monkeypatch.setattr(piperx_demonstration, "home_arms", lambda *args: None)
    monkeypatch.setattr(piperx_demonstration, "ensure_can_control", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        piperx_dynamics,
        "PiperXDynamics",
        lambda _: SimpleNamespace(torques=lambda *args, **kwargs: np.zeros(6)),
    )

    def fail_send(*args, **kwargs) -> None:
        raise OSError("CAN write failed")

    monkeypatch.setattr(leaders[0], "move_mit", fail_send)
    monkeypatch.setattr(mit_control, "disable", lambda arm: pytest.fail("must not disable"))
    demo = piperx_demonstration.PiperXDemonstration(robot, leader)
    with pytest.raises(RuntimeError, match="CAN write failed"), demo.episode():
        pytest.fail("Recording started after MIT send failure.")
    assert all(arm.mode == 1 and arm.js_targets == [[0.08] * 6] for arm in leaders)
    assert all(not arm.js_targets for arm in followers)


def test_joint_positions_ignore_stale_aggregate_timestamp() -> None:
    arm = PiperXArm(PiperArmConfig(port="can_test"), max_feedback_age=0.25)
    now = time.time()
    parser = SimpleNamespace(
        joint_12=SimpleNamespace(msg=object(), timestamp=now),
        joint_34=SimpleNamespace(msg=object(), timestamp=now),
        joint_56=SimpleNamespace(msg=object(), timestamp=now),
    )
    arm.driver = SimpleNamespace(_parser=parser)
    aggregate = SimpleNamespace(msg=[0.1] * 6, timestamp=now - 0.254)

    assert arm._joint_positions(aggregate, "joint") == [0.1] * 6

    parser.joint_34.timestamp = now - 1.0
    with pytest.raises(PiperXFeedbackUnavailable, match="joint_34"):
        arm._joint_positions(aggregate, "joint")


def test_mit_failure_during_episode_homes_all_arms_without_disabling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaders = (FakeArm("left_leader"), FakeArm("right_leader"))
    followers = (FakeArm("left_follower"), FakeArm("right_follower"))
    robot = SimpleNamespace(arms=followers)
    leader = BiPiperXLeader(
        BiPiperXLeaderConfig(
            mit_zero_drag=True,
            mit_gripper_mode="closed",
            home_duration_s=0.02,
        )
    )
    leader.arms = leaders
    leader._resources = ExitStack()
    monkeypatch.setattr(piperx_demonstration, "ensure_can_control", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        piperx_dynamics,
        "PiperXDynamics",
        lambda _: SimpleNamespace(torques=lambda *args, **kwargs: np.zeros(6)),
    )
    monkeypatch.setattr(mit_control, "_JS_HANDOFF_HOLD_S", 0.01)
    monkeypatch.setattr(mit_control, "_MIT_MODE_CONFIRM_S", 0.01)
    monkeypatch.setattr(mit_control, "_MIT_RELEASE_S", 0.01)
    monkeypatch.setattr(mit_control, "disable", lambda arm: pytest.fail("must not disable"))
    home_calls: list[tuple[FakeArm, ...]] = []
    monkeypatch.setattr(
        piperx_demonstration,
        "home_arms",
        lambda arms, duration: home_calls.append(tuple(arms)),
    )

    original_move_mit = leaders[0].move_mit

    def fail_after_release(joint: int, **kwargs: float) -> None:
        original_move_mit(joint, **kwargs)
        if len(leaders[0].mit_ticks) > 18:
            raise OSError("CAN write failed")

    monkeypatch.setattr(leaders[0], "move_mit", fail_after_release)

    demo = piperx_demonstration.PiperXDemonstration(robot, leader)
    with demo.episode():
        deadline = time.monotonic() + 2.0
        while demo._failure is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert demo._failure is not None

    assert demo._failure is not None
    assert home_calls == [(*leaders, *followers), (*leaders, *followers)]
    assert all(arm.mode == 1 and arm.positions == [0.08] * 6 for arm in leaders)


def test_leader_grippers_become_passive_for_feedback_teleop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaders = (FakeArm("left_leader"), FakeArm("right_leader"))
    followers = (FakeArm("left_follower"), FakeArm("right_follower"))
    robot = SimpleNamespace(arms=followers)
    leader = BiPiperXLeader(
        BiPiperXLeaderConfig(
            mit_zero_drag=True,
            mit_gripper_mode="feedback",
            home_duration_s=0.02,
        )
    )
    leader.arms = leaders
    leader._resources = ExitStack()
    leaders[0].gripper_width_m = 0.05
    leaders[1].gripper_width_m = 0.03
    monkeypatch.setattr(piperx_demonstration, "ensure_can_control", lambda *args, **kwargs: False)
    monkeypatch.setattr(piperx_demonstration, "home_arms", lambda *args: None)
    monkeypatch.setattr(
        piperx_dynamics,
        "PiperXDynamics",
        lambda _: SimpleNamespace(torques=lambda *args, **kwargs: np.zeros(6)),
    )
    monkeypatch.setattr(mit_control, "_JS_HANDOFF_HOLD_S", 0.01)
    monkeypatch.setattr(mit_control, "_MIT_MODE_CONFIRM_S", 0.01)
    monkeypatch.setattr(mit_control, "_MIT_RELEASE_S", 0.01)

    demo = piperx_demonstration.PiperXDemonstration(robot, leader)
    with demo.episode():
        pass

    assert all(not arm.gripper_enabled for arm in leaders)
    assert all(arm.disable_gripper_calls == 1 for arm in leaders)
    np.testing.assert_allclose(
        leaders[0].observe_mit(closed_gripper=False)[6],
        0.05 * LEGACY_GRIPPER_PER_METER,
    )
