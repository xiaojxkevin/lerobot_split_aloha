from typing import Any

from torch import Tensor

from lerobot.lerobot_types import PolicyAction
from lerobot.processor import PolicyProcessorPipeline, make_default_pre_post_processors

from .configuration_pi_ws import PiWSConfig


def make_pi_ws_pre_post_processors(
    config: PiWSConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    # D4RT owns normalization and returns absolute commands in the dataset domain.
    return make_default_pre_post_processors(config)
