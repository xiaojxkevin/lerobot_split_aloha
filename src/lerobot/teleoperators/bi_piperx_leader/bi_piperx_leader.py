from contextlib import ExitStack
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.robots.bi_piperx_follower.bi_piperx_follower import PiperArm
from lerobot.robots.bi_piperx_follower.config_bi_piperx_follower import ACTION_KEYS
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_piperx_leader import BiPiperXLeaderConfig


class BiPiperXLeader(Teleoperator):
    """Read two separately connected Piper-X arms already in leader mode."""

    config_class = BiPiperXLeaderConfig
    name = "bi_piperx_leader"

    def __init__(self, config: BiPiperXLeaderConfig) -> None:
        super().__init__(config)
        self.arms = (
            PiperArm(config.left_arm_config, config.feedback_max_age_s),
            PiperArm(config.right_arm_config, config.feedback_max_age_s),
        )
        self._resources: ExitStack | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(ACTION_KEYS, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._resources is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        """Use the leader controller's absolute positions."""

    def configure(self) -> None:
        """Leader mode is configured on the controller before recording."""

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        with ExitStack() as resources:
            for arm in self.arms:
                arm.connect(resources)
            self._resources = resources.pop_all()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        left, right = (arm.observe(leader=True) for arm in self.arms)
        return dict(zip(ACTION_KEYS, left + right, strict=True))

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """The leader only supplies teleoperation targets."""

    @check_if_not_connected
    def disconnect(self) -> None:
        assert self._resources is not None
        try:
            self._resources.close()
        finally:
            self._resources = None
