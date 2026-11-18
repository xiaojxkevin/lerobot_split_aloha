import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))
ARM_ACTION_KEYS = (*(f"{name}.pos" for name in JOINT_NAMES), "gripper.pos")
ACTION_KEYS = tuple(f"{side}_{key}" for side in ("left", "right") for key in ARM_ACTION_KEYS)
LEGACY_GRIPPER_PER_METER = 1000 * math.pi / 180
DEFAULT_CONTROLLER_LIMITS = Path(__file__).resolve().parents[4] / "assets/split-aloha/controller_limits.json"
PiperFirmware = Literal["default", "v183", "v188", "v189"]


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


@RobotConfig.register_subclass("bi_piperx_follower")
@dataclass(kw_only=True)
class BiPiperXFollowerConfig(RobotConfig):
    left_arm_config: PiperArmConfig = field(default_factory=PiperArmConfig)
    right_arm_config: PiperArmConfig = field(default_factory=PiperArmConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    follower_calibration_dir: Path | None = None
    controller_limits_path: Path = DEFAULT_CONTROLLER_LIMITS
    joint_command: Literal["move_js"] = "move_js"
    feedback_max_age_s: float = 0.25
    enable_timeout_s: float = 3.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.joint_command != "move_js":
            raise ValueError("Piper-X joint_command must be 'move_js'.")
        if self.left_arm_config.port is None:
            self.left_arm_config = replace(self.left_arm_config, port="can_left_slave")
        if self.right_arm_config.port is None:
            self.right_arm_config = replace(self.right_arm_config, port="can_right_slave")
        if self.left_arm_config.port == self.right_arm_config.port:
            raise ValueError("The left and right arms require different CAN ports.")
        if set(self.cameras) & set(ACTION_KEYS):
            raise ValueError("Camera names must not overlap joint or gripper feature names.")
        for name in ("feedback_max_age_s", "enable_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")


@RobotConfig.register_subclass("bi_piper_follower")
@dataclass(kw_only=True)
class LegacyBiPiperXFollowerConfig(BiPiperXFollowerConfig):
    """Compatibility with configs saved before the Piper-X rename."""
