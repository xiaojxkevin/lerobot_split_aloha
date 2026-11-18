import math
from dataclasses import dataclass, field, replace

from lerobot.robots.bi_piperx_follower.config_bi_piperx_follower import PiperArmConfig

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_piperx_leader")
@dataclass(kw_only=True)
class BiPiperXLeaderConfig(TeleoperatorConfig):
    left_arm_config: PiperArmConfig = field(default_factory=PiperArmConfig)
    right_arm_config: PiperArmConfig = field(default_factory=PiperArmConfig)
    feedback_max_age_s: float = 0.25

    def __post_init__(self) -> None:
        if self.left_arm_config.port is None:
            self.left_arm_config = replace(self.left_arm_config, port="can_left_main")
        if self.right_arm_config.port is None:
            self.right_arm_config = replace(self.right_arm_config, port="can_right_main")
        if self.left_arm_config.port == self.right_arm_config.port:
            raise ValueError("The left and right leaders require different CAN ports.")
        if not math.isfinite(self.feedback_max_age_s) or self.feedback_max_age_s <= 0:
            raise ValueError("feedback_max_age_s must be finite and positive.")


@TeleoperatorConfig.register_subclass("bi_piper_leader")
@dataclass(kw_only=True)
class LegacyBiPiperXLeaderConfig(BiPiperXLeaderConfig):
    """Compatibility with configs saved before the Piper-X rename."""
