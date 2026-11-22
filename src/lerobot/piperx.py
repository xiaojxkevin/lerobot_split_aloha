"""Shared configuration and single-arm CAN connection for Piper-X."""

import json
import math
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from lerobot.utils.import_utils import _pyagxarm_available

if TYPE_CHECKING or _pyagxarm_available:
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
ARM_ACTION_KEYS = (*(f"{name}.pos" for name in JOINT_NAMES), "gripper.pos")
ACTION_KEYS = tuple(f"{side}_{key}" for side in ("left", "right") for key in ARM_ACTION_KEYS)
LEGACY_GRIPPER_PER_METER = 1000 * math.pi / 180
PiperFirmware = Literal["default", "v183", "v188", "v189"]


class PiperXFeedbackUnavailable(RuntimeError):
    """The SDK has no fresh feedback for a requested signal."""


@dataclass(kw_only=True)
class PiperArmConfig:
    port: str | None = None
    firmware: PiperFirmware | None = None
    speed_percent: int = 100
    gripper_max_width_m: float = 0.1

    def __post_init__(self) -> None:
        if self.port is not None and not self.port.strip():
            raise ValueError("The CAN port must not be empty.")
        if not 1 <= self.speed_percent <= 100:
            raise ValueError("speed_percent must be in [1, 100].")
        if not math.isfinite(self.gripper_max_width_m) or self.gripper_max_width_m <= 0:
            raise ValueError("gripper_max_width_m must be finite and positive.")


class PiperXArm:
    """Single Piper-X CAN connection using absolute joint angles and gripper width."""

    def __init__(self, config: PiperArmConfig, max_feedback_age: float) -> None:
        self.config = config
        self.max_feedback_age = max_feedback_age
        self.driver: Any = None
        self.gripper: Any = None
        self.limits: list[tuple[float, float]] = []
        self._joint_read_lock = Lock()

    def connect(
        self,
        resources: ExitStack,
        calibration_file: Path | None = None,
        firmware_profile: PiperFirmware = "default",
        *,
        init_gripper: bool = True,
        local_loopback: bool = False,
    ) -> None:
        if not _pyagxarm_available:
            raise ImportError("Install pyAgxArm using the Split-ALOHA setup in README.md.")

        config = create_agx_arm_config(
            ArmModel.PIPER_X,
            firmeware_version=self.config.firmware or firmware_profile,
            channel=self.config.port,
            interface="socketcan",
            auto_connect=False,
            local_loopback=local_loopback,
        )
        self.limits = [tuple(config["joint_limits"][f"joint{index}"]) for index in range(1, 7)]
        if calibration_file is not None:
            with calibration_file.open() as file:
                calibration = json.load(file)
            for index, key in enumerate(ARM_ACTION_KEYS[:6]):
                entry = calibration[key]
                if entry["homing_offset"] != 0 or entry["drive_mode"] != 0:
                    raise ValueError(f"{calibration_file}: {key} requires zero offset and drive_mode=0.")
                minimum = math.radians(entry["range_min"] / 1000)
                maximum = math.radians(entry["range_max"] / 1000)
                sdk_min, sdk_max = self.limits[index]
                lower, upper = max(minimum, sdk_min), min(maximum, sdk_max)
                if not math.isfinite(minimum) or not math.isfinite(maximum) or lower >= upper:
                    raise ValueError(f"{calibration_file}: invalid limits for {key}.")
                self.limits[index] = (lower, upper)
        self.driver = AgxArmFactory.create_arm(config)
        resources.callback(self.driver.disconnect)
        self.driver.connect()
        if init_gripper:
            self.gripper = self.driver.init_effector(self.driver.OPTIONS.EFFECTOR.AGX_GRIPPER)

    def enable(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while not self.driver.enable():
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.config.port}: Piper-X did not confirm enable.")
            time.sleep(0.02)
        self.driver.set_speed_percent(self.config.speed_percent)

    def set_gripper_passive(self, timeout_s: float = 1.0) -> None:
        """Disable the gripper motor so its jaws can be moved by hand.

        Only the gripper effector is affected; the six arm motors are left in
        their current state.  A disabled width-mode gripper keeps reporting its
        measured jaw width, which is what teleoperation mirrors.
        """
        if self.gripper is None:
            raise RuntimeError(f"{self.config.port}: gripper is not initialized.")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                status = self._feedback(self.gripper.get_gripper_status(), "gripper")
            except PiperXFeedbackUnavailable:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
                continue
            if not status.foc_status.driver_enable_status:
                return
            self.gripper.disable_gripper()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.config.port}: gripper did not become passive.")
            time.sleep(0.02)

    def _feedback(self, message: Any, name: str) -> Any:
        if message is None or message.msg is None:
            raise PiperXFeedbackUnavailable(f"{self.config.port}: missing {name} feedback.")
        age = time.time() - message.timestamp
        if not 0 <= age <= self.max_feedback_age:
            raise PiperXFeedbackUnavailable(f"{self.config.port}: stale {name} feedback ({age:.3f}s).")
        return message.msg

    def get_joint_positions(self) -> list[float]:
        """Read six fresh measured joint angles."""
        # pyAgxArm reuses a mutable six-angle buffer for every call to get_joint_angles().
        with self._joint_read_lock:
            return self._joint_positions(self.driver.get_joint_angles(), "joint")

    def _joint_positions(self, message: Any, label: str, *, prefix: str = "") -> list[float]:
        if message is None or message.msg is None:
            raise PiperXFeedbackUnavailable(f"{self.config.port}: missing {label} feedback.")
        # The aggregate timestamp belongs to only the last received pair, so it can be
        # older than the other pairs even when the feedback is fresh.  Check every pair
        # instead: each one is an independent 200 Hz frame from the CAN reader thread.
        for pair in ("joint_12", "joint_34", "joint_56"):
            self._feedback(getattr(self.driver._parser, prefix + pair, None), prefix + pair)
        positions = list(map(float, message.msg))
        if len(positions) != len(JOINT_NAMES) or not all(map(math.isfinite, positions)):
            raise RuntimeError(f"{self.config.port}: invalid Piper-X {label} feedback.")
        return positions

    def _gripper_width(self, value: float) -> float:
        width = float(value) * LEGACY_GRIPPER_PER_METER
        if not math.isfinite(width):
            raise RuntimeError(f"{self.config.port}: invalid Piper-X gripper feedback.")
        return width

    def get_status(self) -> Any:
        """Read the current arm status with the same freshness check as angles."""
        return self._feedback(self.driver.get_arm_status(), "arm status")

    def get_joint_motor_state(self, joint: int) -> Any:
        """Read a fresh high-speed motor state for one joint."""
        return self._feedback(self.driver.get_motor_states(joint), f"joint {joint} motor")

    def get_joint_enable_states(self) -> list[bool]:
        """Read each motor's enable bit from fresh driver feedback."""
        return [
            self._feedback(
                self.driver.get_driver_states(joint), f"joint {joint} driver"
            ).foc_status.driver_enable_status
            for joint in range(1, 7)
        ]
