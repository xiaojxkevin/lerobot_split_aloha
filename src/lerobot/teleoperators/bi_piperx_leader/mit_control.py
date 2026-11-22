"""Piper-X host MIT commands and feedback checks shared by the experiment and teleoperation."""

import math
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar, cast, get_args

import numpy as np
from numpy.typing import NDArray

from lerobot.piperx import PiperArmConfig, PiperFirmware, PiperXArm, PiperXFeedbackUnavailable
from lerobot.utils.import_utils import _can_available, _pyagxarm_available, require_package

if TYPE_CHECKING or _can_available:
    import can

if TYPE_CHECKING or _pyagxarm_available:
    from pyAgxArm import ArmModel, resolve_firmware_profile

if TYPE_CHECKING:
    from lerobot.model.piperx_dynamics import PiperXDynamics


_FIRMWARE_CAN_ID = 0x4AF
_FIRMWARE_REPLY_BYTES = 11 * 8
_COMMAND_PERIOD_S = 0.01
_HOME_DURATION_S = 3.0
_MIT_MODE_CONFIRM_S = 0.25
_JS_HANDOFF_HOLD_S = 0.25
_MIT_RELEASE_S = 0.5
_MIT_HOLD_KP = 10.0
_MIT_HOLD_KD = 0.8
_FEEDBACK_RETRY_S = 0.05
_FEEDBACK_POLL_S = 0.005
_Sample = TypeVar("_Sample")


def read_mit_feedback(read: Callable[[], _Sample]) -> _Sample:
    """Allow a short receive gap without using stale feedback."""
    deadline = time.monotonic() + _FEEDBACK_RETRY_S
    while True:
        try:
            return read()
        except PiperXFeedbackUnavailable:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(remaining, _FEEDBACK_POLL_S))


def _read_firmware_version(port: str) -> str:
    require_package("python-can", "can-dep", import_name="can")
    # The SDK get_firmware() also checks periodic feedback FPS. Read the version
    # reply independently so standby controllers can report their firmware.
    try:
        with can.Bus(
            interface="socketcan",
            channel=port,
            can_filters=[{"can_id": _FIRMWARE_CAN_ID, "can_mask": 0x7FF, "extended": False}],
        ) as bus:
            bus.send(
                can.Message(arbitration_id=_FIRMWARE_CAN_ID, data=[1], is_extended_id=False), timeout=0.2
            )
            deadline = time.monotonic() + 3.0
            reply = bytearray()
            while (remaining := deadline - time.monotonic()) > 0:
                frame = bus.recv(timeout=remaining)
                if frame is None:
                    break
                if frame.is_error_frame or frame.is_remote_frame or len(frame.data) != 8:
                    continue
                if frame.data.startswith(b"H-V"):
                    reply.clear()
                elif not reply:
                    continue
                reply.extend(frame.data)
                if len(reply) == _FIRMWARE_REPLY_BYTES:
                    return reply[60:68].decode("ascii")
    except can.CanError as exc:
        raise RuntimeError(f"{port}: firmware query failed: {exc}") from exc
    raise RuntimeError(f"{port}: missing or incomplete firmware reply on CAN ID 0x4AF.")


def firmware_profile(config: PiperArmConfig) -> PiperFirmware:
    if config.port is None:
        raise ValueError("A CAN port is required to query Piper-X firmware.")
    version = _read_firmware_version(config.port)
    profile = resolve_firmware_profile(ArmModel.PIPER_X, version)
    if profile not in get_args(PiperFirmware):
        raise RuntimeError(f"{config.port}: unsupported driver profile {profile} for firmware {version}.")
    print(f"{config.port}: firmware {version}, driver {profile}", flush=True)
    return cast(PiperFirmware, profile)


def check_status(
    arm: PiperXArm, *, expected_motion_mode: int | None = None, require_can_control: bool = False
) -> None:
    status = arm.get_status()
    modes = arm.driver.ARM_STATUS.CtrlMode
    if status.ctrl_mode not in (modes.STANDBY, modes.CAN_CTRL):
        raise RuntimeError(
            f"{arm.config.port}: control mode {status.ctrl_mode}. Set the controller to host CAN mode."
        )
    if status.arm_status != arm.driver.ARM_STATUS.ArmStatus.NORMAL:
        raise RuntimeError(f"{arm.config.port}: controller reports {status.arm_status}.")
    if require_can_control and status.ctrl_mode != modes.CAN_CTRL:
        raise RuntimeError(f"{arm.config.port}: lost CAN control during motion mode handoff.")
    if expected_motion_mode is not None and (
        status.ctrl_mode != modes.CAN_CTRL or status.mode_feedback != expected_motion_mode
    ):
        raise RuntimeError(
            f"{arm.config.port}: expected motion mode {expected_motion_mode}, got {status.mode_feedback}."
        )


def check_enabled(arm: PiperXArm, phase: str) -> None:
    disabled = [f"J{joint}" for joint, enabled in enumerate(arm.get_joint_enable_states(), 1) if not enabled]
    if disabled:
        raise RuntimeError(f"{arm.config.port}: motor enable lost during {phase}: {', '.join(disabled)}.")


def mit_torques(
    model: "PiperXDynamics",
    positions: list[float],
    width: float,
    limit: float,
    scale: float,
    *,
    j6_scale: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    torque = model.torques(positions, gripper_width_m=width)
    if not np.isfinite(torque).all():
        raise RuntimeError("Non-finite compensation torque.")
    feedforward = scale * torque
    feedforward[5] *= j6_scale
    return torque, np.clip(feedforward, -limit, limit)


def send_mit(
    arm: PiperXArm, targets: list[float], feedforward: NDArray[np.float64], hold_fraction: float
) -> None:
    driver = arm.driver
    driver.set_motion_mode(driver.OPTIONS.MOTION_MODE.MIT)
    for joint, (position, value) in enumerate(zip(targets, feedforward, strict=True), 1):
        driver.move_mit(
            joint,
            p_des=position,
            kp=_MIT_HOLD_KP * hold_fraction,
            kd=_MIT_HOLD_KD * hold_fraction,
            t_ff=float(value),
        )


class MITDrag:
    """Hold the handoff pose in MIT and release to gravity feedforward."""

    def __init__(
        self,
        arm: PiperXArm,
        model: "PiperXDynamics",
        width: float,
        limit: float,
        scale: float,
        j6_scale: float,
    ) -> None:
        self.arm = arm
        self.model = model
        self.width = width
        self.limit = limit
        self.scale = scale
        self.j6_scale = j6_scale
        lower, upper = np.asarray(arm.limits).T
        self.hold_positions = np.clip(arm.get_joint_positions(), lower, upper).tolist()
        arm.driver.set_auto_set_motion_mode_enabled(False)
        self.mode_deadline = time.monotonic() + _MIT_MODE_CONFIRM_S
        self.release_started_at: float | None = None

    def tick(self, now: float, positions: list[float]) -> bool:
        require_mit = now >= self.mode_deadline
        mit_mode = self.arm.driver.ARM_STATUS.ModeFeedback.MOVE_MIT
        check_status(self.arm, expected_motion_mode=mit_mode if require_mit else None)
        check_enabled(self.arm, "MIT")
        if require_mit and self.release_started_at is None:
            self.release_started_at = now
        hold = 1.0
        if self.release_started_at is not None:
            hold = max(0.0, 1.0 - (now - self.release_started_at) / _MIT_RELEASE_S)
        _, feedforward = mit_torques(
            self.model, positions, self.width, self.limit, self.scale, j6_scale=self.j6_scale
        )
        send_mit(self.arm, self.hold_positions, feedforward, hold)
        return hold == 0.0


def home_arms(arms: Sequence[PiperXArm], duration_s: float) -> None:
    """Hold each arm at its own measured pose in JS, then interpolate to joint zero."""
    initial_targets = []
    for arm in arms:
        if not all(lower <= 0 <= upper for lower, upper in arm.limits):
            raise RuntimeError(f"{arm.config.port}: joint limits exclude the zero pose.")
        check_status(arm)
        check_enabled(arm, "homing")
        lower, upper = np.asarray(arm.limits).T
        initial_targets.append(np.clip(arm.get_joint_positions(), lower, upper))
    for arm in arms:
        arm.driver.set_auto_set_motion_mode_enabled(True)

    started_at = time.monotonic()
    while True:
        tick = time.monotonic()
        elapsed = tick - started_at
        homing_elapsed = elapsed - _JS_HANDOFF_HOLD_S
        remaining = min(1.0, max(0.0, 1.0 - homing_elapsed / duration_s))
        max_error = 0.0
        for arm, initial in zip(arms, initial_targets, strict=True):
            js_mode = arm.driver.ARM_STATUS.ModeFeedback.MOVE_J
            check_status(arm, expected_motion_mode=js_mode if homing_elapsed >= 0 else None)
            check_enabled(arm, "homing")
            max_error = max(max_error, max(map(abs, arm.get_joint_positions())))
            arm.driver.move_js((initial * remaining).tolist())
        if remaining == 0:
            if max_error <= math.radians(3.0):
                return
            if homing_elapsed >= duration_s + 2.0:
                raise RuntimeError(
                    f"Piper-X homing timed out with {math.degrees(max_error):.1f}° joint error."
                )
        time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - tick)))


def disable(arm: PiperXArm) -> None:
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        arm.driver.disable()
        try:
            if not any(arm.get_joint_enable_states()):
                return
        except RuntimeError:
            pass
        time.sleep(0.02)
    raise RuntimeError(f"{arm.config.port}: motor disable was not confirmed.")
