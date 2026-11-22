import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from lerobot.cameras import CameraConfig
from lerobot.piperx import (
    ACTION_KEYS,
    ARM_ACTION_KEYS as ARM_ACTION_KEYS,
    JOINT_NAMES as JOINT_NAMES,
    LEGACY_GRIPPER_PER_METER as LEGACY_GRIPPER_PER_METER,
    PiperArmConfig as PiperArmConfig,
    PiperFirmware as PiperFirmware,
)

from ..config import RobotConfig

DEFAULT_CONTROLLER_LIMITS = Path(__file__).resolve().parents[4] / "assets/split-aloha/controller_limits.json"


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
