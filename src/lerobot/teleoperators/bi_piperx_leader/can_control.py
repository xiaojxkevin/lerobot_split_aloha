"""Select host CAN control for a connected Piper-X leader."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from lerobot.utils.import_utils import _can_available, require_package

if TYPE_CHECKING:
    from .bi_piperx_leader import PiperXLeaderArm

if TYPE_CHECKING or _can_available:
    import can

_MODE_CAN_ID = 0x151
_FEEDBACK_TIMEOUT_S = 2.0


def _wait_for_status(
    arm: PiperXLeaderArm, expected_mode: int | None = None, *, sample: Callable[[Any], None] | None = None
) -> Any:
    deadline = time.monotonic() + _FEEDBACK_TIMEOUT_S
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status = arm.get_status()
        except RuntimeError as exc:
            last_error = str(exc)
            if sample is not None:
                sample(None)
        else:
            if status.arm_status != arm.driver.ARM_STATUS.ArmStatus.NORMAL:
                raise RuntimeError(f"controller reports {status.arm_status}")
            if sample is not None:
                sample(status.ctrl_mode)
            if expected_mode is None or status.ctrl_mode == expected_mode:
                return status
            last_error = f"expected {expected_mode}, controller still reports {status.ctrl_mode}"
        time.sleep(0.005 if sample is not None else 0.02)
    raise RuntimeError(last_error or "timed out waiting for controller status")


def _request_mode(
    arm: PiperXLeaderArm,
    bus: can.BusABC,
    mode: int,
    *,
    operator_confirmed: bool,
    sample: Callable[[Any], None] | None,
) -> None:
    deadline = time.monotonic() + _FEEDBACK_TIMEOUT_S
    while True:
        try:
            enabled = arm.get_joint_enable_states()
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
        else:
            if any(enabled) and not operator_confirmed:
                raise RuntimeError("disable all six motors before switching control mode")
            break
    # Set control mode and select MOVE_J at zero command speed. No joint target
    # or motor-enable command is sent during this transition.
    bus.send(
        can.Message(arbitration_id=_MODE_CAN_ID, data=[mode, 1, 0, 0, 0, 0, 0, 0], is_extended_id=False),
        timeout=0.2,
    )
    _wait_for_status(arm, mode, sample=sample)


def ensure_can_control(arm: PiperXLeaderArm, *, operator_confirmed: bool = False) -> bool:
    """Confirm CAN control and return whether a mode change was made."""
    status = _wait_for_status(arm)
    modes = arm.driver.ARM_STATUS.CtrlMode
    if status.ctrl_mode == modes.CAN_CTRL:
        return False

    if status.ctrl_mode == modes.TEACHING_MODE:
        teaching = arm.driver.ARM_STATUS.TeachingState
        if status.teach_status not in (teaching.DISABLED, teaching.STOP_RECORDING):
            raise RuntimeError(f"stop teaching first (controller reports {status.teach_status})")
    elif status.ctrl_mode != modes.STANDBY:
        raise RuntimeError(f"expected Teaching or Standby, controller reports {status.ctrl_mode}")

    require_package("python-can", "can-dep", import_name="can")
    port = arm.config.port
    if port is None:
        raise ValueError("A CAN port is required to switch control mode.")
    initial_positions = arm.get_joint_positions() if operator_confirmed else None
    observations: list[tuple[str, tuple[bool, ...]]] = []
    max_motion_deg = 0.0
    missing_samples = 0

    def observe(mode: Any) -> None:
        nonlocal max_motion_deg, missing_samples
        assert initial_positions is not None
        if mode is None:
            missing_samples += 1
            return
        try:
            enabled = tuple(arm.get_joint_enable_states())
            positions = arm.get_joint_positions()
        except RuntimeError:
            missing_samples += 1
            return
        observation = (str(mode), enabled)
        if not observations or observation != observations[-1]:
            observations.append(observation)
        motion_deg = max(
            abs(math.degrees(after - before))
            for before, after in zip(initial_positions, positions, strict=True)
        )
        max_motion_deg = max(max_motion_deg, motion_deg)

    sample = observe if operator_confirmed else None
    if sample is not None:
        sample(status.ctrl_mode)
    try:
        with can.Bus(interface="socketcan", channel=port) as bus:
            if status.ctrl_mode == modes.TEACHING_MODE:
                _request_mode(arm, bus, modes.STANDBY, operator_confirmed=operator_confirmed, sample=sample)
            _request_mode(arm, bus, modes.CAN_CTRL, operator_confirmed=operator_confirmed, sample=sample)
    except can.CanError as exc:
        raise RuntimeError(f"CAN mode request failed: {exc}") from exc
    finally:
        if sample is not None:
            for mode, enabled in observations:
                print(f"{port}: {mode}, motor enable J1–J6: {list(enabled)}", flush=True)
            displacement = f"{max_motion_deg:.3f}°" if observations else "unavailable"
            print(
                f"{port}: maximum sampled joint displacement {displacement}, "
                f"missing feedback samples: {missing_samples}",
                flush=True,
            )
    return True
