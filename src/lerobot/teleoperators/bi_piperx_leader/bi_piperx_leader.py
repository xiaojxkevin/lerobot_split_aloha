from contextlib import ExitStack
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.piperx import ACTION_KEYS, PiperXArm
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_bi_piperx_leader import BiPiperXLeaderConfig
from .mit_control import firmware_profile, read_mit_feedback


class PiperXLeaderArm(PiperXArm):
    """Read the teleoperation commands transmitted by a Piper-X leader."""

    def get_leader_joint_positions(self) -> list[float]:
        return self._joint_positions(self.driver.get_leader_joint_angles(), "leader joint", prefix="leader_")

    def observe(self) -> list[float]:
        gripper = self._feedback(self.gripper.get_gripper_ctrl_states(), "leader gripper")
        if gripper.status_code != 1:
            raise RuntimeError(f"{self.config.port}: expected a leader gripper width command (status 1).")
        width = self._gripper_width(gripper.value)
        return [*self.get_leader_joint_positions(), width]

    def observe_mit(self, *, closed_gripper: bool) -> list[float]:
        joints = self.get_joint_positions()
        if closed_gripper:
            return [*joints, 0.0]
        gripper = self._feedback(self.gripper.get_gripper_status(), "MIT teach pendant width")
        if gripper.mode != "width":
            raise RuntimeError(f"{self.config.port}: expected width-mode teach pendant feedback in MIT.")
        return [*joints, self._gripper_width(gripper.value)]


class BiPiperXLeader(Teleoperator):
    """Read two separately connected Piper-X arms already in leader mode."""

    config_class = BiPiperXLeaderConfig
    name = "bi_piperx_leader"

    def __init__(self, config: BiPiperXLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.arms = (
            PiperXLeaderArm(config.left_arm_config, config.feedback_max_age_s),
            PiperXLeaderArm(config.right_arm_config, config.feedback_max_age_s),
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
                if self.config.mit_zero_drag:
                    arm.connect(
                        resources, firmware_profile=arm.config.firmware or firmware_profile(arm.config)
                    )
                else:
                    arm.connect(resources)
            self._resources = resources.pop_all()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        if self.config.mit_zero_drag:
            return read_mit_feedback(self._read_mit_action)
        left, right = (arm.observe() for arm in self.arms)
        return dict(zip(ACTION_KEYS, left + right, strict=True))

    def _read_mit_action(self) -> RobotAction:
        closed_gripper = self.config.mit_gripper_mode == "closed"
        left, right = (arm.observe_mit(closed_gripper=closed_gripper) for arm in self.arms)
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
