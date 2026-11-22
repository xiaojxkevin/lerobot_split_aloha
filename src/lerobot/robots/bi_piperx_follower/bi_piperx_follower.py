import math
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.piperx import ACTION_KEYS, LEGACY_GRIPPER_PER_METER, PiperXArm
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_bi_piperx_follower import BiPiperXFollowerConfig
from .controller_limits import PiperControllerLimits


class PiperArm(PiperXArm):
    """Read measured follower feedback and send follower joint targets."""

    def observe(self) -> list[float]:
        gripper = self._feedback(self.gripper.get_gripper_status(), "gripper")
        if gripper.mode != "width":
            raise RuntimeError(f"{self.config.port}: expected width-mode gripper feedback.")
        width = self._gripper_width(gripper.value)
        return [*self.get_joint_positions(), width]

    def get_replay_feedback(self) -> dict[str, Any]:
        """Read optional diagnostic samples from the SDK's receive cache."""
        motor_velocity = []
        for index in range(1, 7):
            message = self.driver.get_motor_states(index)
            try:
                state = self._feedback(message, f"motor {index}")
                timestamp = float(message.timestamp)
                velocity = float(state.velocity)
            except RuntimeError:
                motor_velocity.append(None)
                continue
            motor_velocity.append(
                {"rad_s": velocity, "timestamp_unix_s": timestamp}
                if math.isfinite(velocity) and timestamp == message.timestamp
                else None
            )

        flange_pose = None
        try:
            # The SDK assembles the pose from three CAN frames. A partial pose can
            # contain default zeros, so validate all three parts independently.
            parts = {
                part: getattr(self.driver._parser, f"end_pose_{part}", None) for part in ("xy", "zrx", "ryrz")
            }
            timestamps = {}
            for part, message in parts.items():
                self._feedback(message, f"flange pose {part}")
                timestamps[part] = float(message.timestamp)
            message = self.driver.get_flange_pose()
            pose = list(map(float, self._feedback(message, "flange pose")))
            if (
                len(pose) == 6
                and all(map(math.isfinite, pose))
                and all(timestamps[part] == frame.timestamp for part, frame in parts.items())
            ):
                flange_pose = {
                    "xyz_m": pose[:3],
                    "rpy_rad": pose[3:],
                    "timestamp_unix_s": max(timestamps.values()),
                    "part_timestamps_unix_s": timestamps,
                }
        except RuntimeError:
            pass
        return {"motor_velocity": motor_velocity, "flange_pose": flange_pose}

    def limit_target(self, target: list[float]) -> list[float]:
        joints = [
            min(upper, max(lower, value))
            for value, (lower, upper) in zip(target[:6], self.limits, strict=True)
        ]
        gripper_limit = self.config.gripper_max_width_m * LEGACY_GRIPPER_PER_METER
        gripper = min(gripper_limit, max(0.0, target[6]))
        return [*joints, gripper]

    def send(self, target: list[float]) -> None:
        self.driver.move_js(target[:6])
        self.gripper.move_gripper_m(target[6] / LEGACY_GRIPPER_PER_METER)


class BiPiperXFollower(Robot):
    """Split-ALOHA dual Piper-X follower with D4RT-compatible joint14 ordering."""

    config_class = BiPiperXFollowerConfig
    name = "bi_piperx_follower"

    def __init__(self, config: BiPiperXFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.arms = (
            PiperArm(config.left_arm_config, config.feedback_max_age_s),
            PiperArm(config.right_arm_config, config.feedback_max_age_s),
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._resources: ExitStack | None = None
        self._executor: ThreadPoolExecutor | None = None

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {
            **self.action_features,
            **{name: (camera.height, camera.width, 3) for name, camera in self.cameras.items()},
        }

    @property
    def is_connected(self) -> bool:
        return self._resources is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """PyAgxArm provides absolute joint coordinates."""

    def configure(self) -> None:
        """The SDK sets high-follow mode when dispatching `move_js`."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        limits = PiperControllerLimits.load(self.config.controller_limits_path)
        with ExitStack() as resources:
            for side, arm in zip(("left", "right"), self.arms, strict=True):
                directory = self.config.follower_calibration_dir
                calibration_file = directory / f"follower_{side}.json" if directory is not None else None
                arm.connect(resources, calibration_file, limits.firmware_profile)
            for arm in self.arms:
                limits.apply(arm.driver, arm.config.port)
            for camera in self.cameras.values():
                camera.connect()
                resources.callback(camera.disconnect)
            for arm in self.arms:
                arm.enable(self.config.enable_timeout_s)
            self._executor = resources.enter_context(ThreadPoolExecutor(max_workers=2))
            self._resources = resources.pop_all()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left, right = (arm.observe() for arm in self.arms)
        return {
            **dict(zip(ACTION_KEYS, left + right, strict=True)),
            **{name: camera.read_latest() for name, camera in self.cameras.items()},
        }

    @check_if_not_connected
    def get_replay_feedback(self) -> dict[str, dict[str, Any]]:
        return {
            side: arm.get_replay_feedback() for side, arm in zip(("left", "right"), self.arms, strict=True)
        }

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if set(action) != set(ACTION_KEYS):
            raise ValueError("Expected a complete left/right Piper-X joint14 action.")
        ordered = [float(action[key]) for key in ACTION_KEYS]
        if not all(map(math.isfinite, ordered)):
            raise ValueError("Piper-X action must be finite.")
        targets = [
            arm.limit_target(target)
            for arm, target in zip(self.arms, (ordered[:7], ordered[7:]), strict=True)
        ]
        assert self._executor is not None
        pending = [
            self._executor.submit(arm.send, target) for arm, target in zip(self.arms, targets, strict=True)
        ]
        wait(pending)
        for result in pending:
            result.result()
        return dict(zip(ACTION_KEYS, targets[0] + targets[1], strict=True))

    @check_if_not_connected
    def disconnect(self) -> None:
        assert self._resources is not None
        try:
            self._resources.close()
        finally:
            self._resources = None
            self._executor = None
