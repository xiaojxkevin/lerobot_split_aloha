"""Episode lifecycle for Split-ALOHA host MIT teleoperation."""

import logging
import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from threading import Event, Thread

from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.piperx import PiperXArm
from lerobot.robots.bi_piperx_follower import BiPiperXFollower, BiPiperXFollowerConfig
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.bi_piperx_leader import BiPiperXLeader, BiPiperXLeaderConfig
from lerobot.teleoperators.bi_piperx_leader.can_control import ensure_can_control
from lerobot.teleoperators.bi_piperx_leader.mit_control import (
    _COMMAND_PERIOD_S,
    MITDrag,
    home_arms,
    read_mit_feedback,
)
from lerobot.teleoperators.config import TeleoperatorConfig

logger = logging.getLogger(__name__)

# A failed MIT worker switches the leaders to move_js at their last measured pose.
# The episode keeps running, and this is how long the episode exit keeps retrying
# the final return to zero before giving up (without disabling the motors).
_RECOVERY_RETRY_S = 5.0
_RECOVERY_POLL_S = 0.2


def uses_piperx_demonstration(
    robot: RobotConfig, teleop: TeleoperatorConfig | None, *, dataset: DatasetRecordConfig | None = None
) -> bool:
    if not isinstance(teleop, BiPiperXLeaderConfig) or not teleop.mit_zero_drag:
        return False
    if not isinstance(robot, BiPiperXFollowerConfig):
        raise ValueError("Piper-X MIT demonstration requires a bi_piperx_follower.")
    ports = {
        arm.port
        for arm in (
            teleop.left_arm_config,
            teleop.right_arm_config,
            robot.left_arm_config,
            robot.right_arm_config,
        )
    }
    if len(ports) != 4:
        raise ValueError("Piper-X MIT demonstration requires four separate CAN interfaces.")
    if dataset is not None:
        cameras = [
            camera
            for camera in robot.cameras.values()
            if isinstance(camera, RealSenseCameraConfig) and camera.use_rgb
        ]
        if len(robot.cameras) != 3 or len(cameras) != 3:
            raise ValueError("Piper-X MIT recording requires three RealSense RGB cameras.")
        if len({camera.serial_number_or_name for camera in cameras}) != 3:
            raise ValueError("Piper-X MIT recording requires three distinct RealSense devices.")
        if (
            not dataset.repo_id.strip()
            or not dataset.single_task.strip()
            or not math.isfinite(dataset.episode_time_s)
            or dataset.episode_time_s <= 0
            or dataset.num_episodes <= 0
        ):
            raise ValueError(
                "Piper-X MIT recording requires a repo_id, task, positive episode duration and count."
            )
        if not math.isfinite(dataset.reset_time_s) or dataset.reset_time_s < 0:
            raise ValueError("Piper-X MIT recording requires a finite nonnegative reset duration.")
    return True


class PiperXDemonstration:
    def __init__(self, robot: BiPiperXFollower, leader: BiPiperXLeader) -> None:
        from lerobot.model.piperx_dynamics import PiperXDynamics

        self.robot = robot
        self.leader = leader
        self._model = PiperXDynamics("teach-pendant")
        self._enabled = False
        self._stop = Event()
        self._ready = Event()
        self._failure: Exception | None = None

    def _run_mit(self) -> None:
        drags: list[MITDrag] = []
        last_positions: list[list[float]] | None = None
        try:
            drags = [MITDrag(arm, self._model, 0.0, 8.0, 1.0, 1.0) for arm in self.leader.arms]

            def read_positions() -> list[list[float]]:
                return [drag.arm.get_joint_positions() for drag in drags]

            while not self._stop.is_set():
                now = time.monotonic()
                positions = read_mit_feedback(read_positions)
                last_positions = positions
                if self._stop.is_set():
                    break
                tick = time.monotonic()
                released = [drag.tick(tick, joints) for drag, joints in zip(drags, positions, strict=True)]
                if all(released):
                    self._ready.set()
                self._stop.wait(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - now)))
        except Exception as exc:
            # Do not leave the leaders in MIT with no further commands.  Holding the
            # last measured pose in move_js is safe and leaves them ready for the final
            # return to zero; the recording loop is deliberately not aborted here.
            if last_positions is not None:
                self._hold_last_pose(drags, last_positions)
            self._failure = exc
            self._ready.set()

    def _hold_last_pose(self, drags: Sequence[MITDrag], positions: Sequence[Sequence[float]]) -> None:
        for drag, joints in zip(drags, positions, strict=True):
            try:
                target = [
                    min(max(float(value), lower), upper)
                    for value, (lower, upper) in zip(joints, drag.arm.limits, strict=True)
                ]
                drag.arm.driver.set_auto_set_motion_mode_enabled(True)
                drag.arm.driver.move_js(target)
            except Exception:
                logger.exception(
                    "%s: failed to hold the last measured pose in move_js.", drag.arm.config.port
                )

    def _return_to_zero(self, arms: Sequence[PiperXArm], failure: Exception | None) -> None:
        if failure is not None:
            logger.error("Piper-X MIT control stopped (%s). Returning all four arms to joint zero.", failure)
        else:
            logger.info("Returning all four Piper-X arms to joint zero in move_js.")
        deadline = time.monotonic() + _RECOVERY_RETRY_S
        while True:
            try:
                home_arms(arms, self.leader.config.home_duration_s)
            except Exception as exc:
                if time.monotonic() >= deadline:
                    logger.exception("Could not return the Piper-X arms to joint zero.")
                    return
                logger.warning("Failed to return the Piper-X arms to joint zero: %s; retrying.", exc)
                time.sleep(_RECOVERY_POLL_S)
            else:
                logger.info("All four Piper-X arms are at joint zero in move_js.")
                return

    @contextmanager
    def episode(self) -> Iterator[None]:
        arms = (*self.leader.arms, *self.robot.arms)
        self._failure = None
        ready_to_home = False
        worker: Thread | None = None
        try:
            if not self._enabled:
                for arm in self.leader.arms:
                    ensure_can_control(arm, operator_confirmed=True)
                    arm.enable(timeout_s=3.0)
            self._enabled = True
            ready_to_home = True
            logger.info("Homing all four Piper-X arms to joint zero in move_js.")
            home_arms(arms, self.leader.config.home_duration_s)
            if self.leader.config.mit_gripper_mode == "feedback":
                for arm in self.leader.arms:
                    arm.set_gripper_passive()
                logger.info(
                    "Leader grippers are passive; hand-driven jaw width is mirrored to the followers."
                )
            self._stop.clear()
            self._ready.clear()
            worker = Thread(target=self._run_mit, name="piperx-mit", daemon=True)
            worker.start()
            if not self._ready.wait(timeout=3.0):
                raise RuntimeError("Timed out waiting for the two Piper-X leaders to enter MIT zero-drag.")
            if self._failure is not None:
                raise RuntimeError(f"Piper-X MIT control stopped: {self._failure}") from self._failure
            # Verify pendant input in the actual control mode before collecting any frames.
            try:
                self.leader.get_action()
            except RuntimeError as exc:
                raise RuntimeError(f"Piper-X MIT input unavailable: {exc}") from exc
            logger.info("Both leaders are in MIT zero-drag. Followers are in move_js.")
            yield
        except BaseException:
            self._enabled = False
            raise
        finally:
            self._stop.set()
            if worker is not None:
                worker.join()
            if ready_to_home:
                self._return_to_zero(arms, self._failure)

    @staticmethod
    def wait_for_scene_reset(events: dict[str, bool], duration_s: float) -> None:
        deadline = time.monotonic() + duration_s
        while not events["exit_early"] and time.monotonic() < deadline:
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        events["exit_early"] = False
