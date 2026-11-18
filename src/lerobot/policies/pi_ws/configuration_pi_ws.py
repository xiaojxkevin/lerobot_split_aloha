import math
from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim.optimizers import OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.robots.bi_piperx_follower.config_bi_piperx_follower import ACTION_KEYS
from lerobot.utils.constants import ACTION, OBS_STATE


@dataclass
@PreTrainedConfig.register_subclass("pi_ws")
class PiWSConfig(PreTrainedConfig):
    """Client settings for the D4RT joint-action OpenPI server."""

    action_feature_names: ClassVar[tuple[str, ...]] = ACTION_KEYS
    normalization_mapping: ClassVar[dict[str, NormalizationMode]] = {
        "VISUAL": NormalizationMode.IDENTITY,
        "STATE": NormalizationMode.IDENTITY,
        "ACTION": NormalizationMode.IDENTITY,
    }

    endpoint: str = "ws://localhost:8888"
    device: str | None = "cpu"
    chunk_size: int = 50
    n_action_steps: int = 16
    rtc_training_max_delay: int = 10
    request_timeout_s: float = 5.0
    camera_name_map: dict[str, str] = field(
        default_factory=lambda: {
            "fixed_front": "cam_high",
            "left_arm": "cam_left_wrist",
            "right_arm": "cam_right_wrist",
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.endpoint.startswith(("ws://", "wss://")):
            raise ValueError("PiWS endpoint must start with ws:// or wss://.")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be within the chunk.")
        if not 0 <= self.rtc_training_max_delay < self.chunk_size:
            raise ValueError("rtc_training_max_delay must be smaller than the chunk.")
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be finite and positive.")
        if not self.camera_name_map or len(set(self.camera_name_map.values())) != len(self.camera_name_map):
            raise ValueError("PiWS camera mappings must be nonempty and unique.")
        self.validate_features()

    def validate_features(self) -> None:
        self.input_features = {
            OBS_STATE: PolicyFeature(FeatureType.STATE, (14,)),
            **{
                f"observation.images.{camera}": PolicyFeature(FeatureType.VISUAL, (3, 224, 224))
                for camera in self.camera_name_map
            },
        }
        self.output_features = {ACTION: PolicyFeature(FeatureType.ACTION, (14,))}

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> None:
        return None

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self) -> OptimizerConfig:
        raise NotImplementedError("PiWS is an inference client.")

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        raise NotImplementedError("PiWS is an inference client.")
