"""Home one Piper-X arm and apply MIT gravity compensation for manual dragging."""

import argparse
import csv
import math
import sys
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from statistics import median
from threading import Event
from typing import TextIO, get_args

import numpy as np

from lerobot.model.piperx_dynamics import PiperXDynamics, PiperXTool
from lerobot.piperx import PiperArmConfig
from lerobot.teleoperators.bi_piperx_leader.bi_piperx_leader import PiperXLeaderArm
from lerobot.teleoperators.bi_piperx_leader.can_control import ensure_can_control
from lerobot.teleoperators.bi_piperx_leader.mit_control import (
    _COMMAND_PERIOD_S,
    _HOME_DURATION_S,
    _JS_HANDOFF_HOLD_S,
    _MIT_HOLD_KD,
    _MIT_HOLD_KP,
    _MIT_MODE_CONFIRM_S,
    MITDrag,
    check_enabled,
    check_status,
    disable,
    firmware_profile,
    mit_torques,
    send_mit,
)
from lerobot.utils.keyboard_input import TerminalKeyListener

_J6_STATIC_TARGETS_DEG = (0, 45, 0, -45, 0)
_J6_STATIC_MOVE_S = 2.0
_J6_STATIC_SETTLE_S = 1.0
_J6_STATIC_SAMPLE_S = 2.0
_J6_STATIC_MAX_SPEED_RAD_S = 0.02


def _home(
    arm: PiperXLeaderArm,
    duration_s: float,
    *,
    initial_positions: list[float] | None = None,
    expected_motion_mode: int | None = None,
) -> None:
    if initial_positions is None:
        initial_positions = arm.get_joint_positions()
    lower, upper = np.asarray(arm.limits).T
    initial_target = np.clip(initial_positions, lower, upper)
    started_at = time.monotonic()
    while True:
        start = time.monotonic()
        check_status(arm, expected_motion_mode=expected_motion_mode)
        check_enabled(arm, "homing")
        joints = arm.get_joint_positions()
        remaining_fraction = max(0.0, 1.0 - (start - started_at) / duration_s)
        arm.driver.move_js((initial_target * remaining_fraction).tolist())
        if remaining_fraction == 0.0:
            angles = np.round(np.rad2deg(joints), 2).tolist()
            print(f"Homing trajectory complete. Measured J1-J6 (deg): {angles}.", flush=True)
            return
        time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - start)))


def _drag(
    arm: PiperXLeaderArm,
    model: PiperXDynamics,
    width: float,
    limit: float,
    scale: float,
    *,
    j6_scale: float,
    drag_ready: Event,
    return_requested: Event,
) -> list[float]:
    drag = MITDrag(arm, model, width, limit, scale, j6_scale)
    print(
        f"Switching to MIT at the measured pose with position hold. J6 feedforward multiplier: {j6_scale:g}.",
        flush=True,
    )
    while True:
        start = time.monotonic()
        positions = arm.get_joint_positions()
        if return_requested.is_set():
            return positions
        if drag.tick(start, positions) and not drag_ready.is_set():
            print(
                f"MIT zero-drag active. Measured J6: {math.degrees(positions[5]):+.2f} deg. "
                "Release the arm and press the Left Arrow key to switch to move_js and home. "
                "Support the arm before stopping with Ctrl+C.",
                flush=True,
            )
            drag_ready.set()
        time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - start)))


def _return_to_js(arm: PiperXLeaderArm, positions: list[float], duration_s: float) -> None:
    driver = arm.driver
    js_mode = driver.ARM_STATUS.ModeFeedback.MOVE_J
    lower, upper = np.asarray(arm.limits).T
    targets = np.clip(positions, lower, upper).tolist()
    peak_displacement = np.zeros(6)
    driver.set_auto_set_motion_mode_enabled(True)
    started_at = time.monotonic()
    start = started_at
    # move_js sets JS mode and immediately sends the captured position.
    driver.move_js(targets)
    while True:
        time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - start)))
        start = time.monotonic()
        hold_complete = start - started_at >= _JS_HANDOFF_HOLD_S
        check_status(arm, expected_motion_mode=js_mode if hold_complete else None, require_can_control=True)
        check_enabled(arm, "MIT to move_js handoff")
        measured = arm.get_joint_positions()
        peak_displacement = np.maximum(peak_displacement, np.abs(np.subtract(measured, positions)))
        if hold_complete:
            break
        driver.move_js(targets)
    handoff_deg = np.round(np.rad2deg(positions), 2).tolist()
    peak_deg = np.round(np.rad2deg(peak_displacement), 2).tolist()
    print(
        f"move_js confirmed (MOVE_J feedback), all motors enabled. "
        f"Handoff J1-J6 (deg): {handoff_deg}. Peak displacement during handoff (deg): {peak_deg}.",
        flush=True,
    )
    print(f"Homing via move_js over {duration_s:g} s at 100 Hz.", flush=True)
    _home(arm, duration_s, initial_positions=targets, expected_motion_mode=js_mode)
    print("move_js holding the zero target. Support the arm before stopping with Ctrl+C.", flush=True)
    while True:
        start = time.monotonic()
        check_status(arm, expected_motion_mode=js_mode)
        check_enabled(arm, "move_js zero hold")
        arm.get_joint_positions()
        driver.move_js([0.0] * 6)
        time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - start)))


def _calibrate_j6(
    arm: PiperXLeaderArm,
    model: PiperXDynamics,
    width: float,
    limit: float,
    scale: float,
    output: TextIO,
    *,
    j6_scale: float,
) -> None:
    """Sample J6 holding effort in MIT while J1-J5 hold the handoff pose."""
    writer = csv.writer(output)
    writer.writerow(
        [
            "elapsed_s",
            "pose_index",
            "j6_target_deg",
            *(f"j{joint}_measured_deg" for joint in range(1, 7)),
            "j6_error_deg",
            "j6_velocity_rad_s",
            "j6_current_a",
            "j6_model_nm",
            "j6_sdk_feedforward_nm",
            "j6_pd_estimate_nm",
            "j6_total_estimate_nm",
            "stationary",
        ]
    )
    output.flush()
    driver = arm.driver
    driver.set_auto_set_motion_mode_enabled(False)
    mit_mode = driver.ARM_STATUS.ModeFeedback.MOVE_MIT
    lower, upper = np.asarray(arm.limits).T
    targets = np.clip(arm.get_joint_positions(), lower, upper).tolist()
    started_at = time.monotonic()
    pose_started_at = started_at + _MIT_MODE_CONFIRM_S
    pose_index = 0
    initial_j6 = targets[5]
    summaries: list[tuple[float, float, float, float]] = []
    sampling_at = _J6_STATIC_MOVE_S + _J6_STATIC_SETTLE_S
    pose_duration = sampling_at + _J6_STATIC_SAMPLE_S
    print(
        f"Switching to MIT with position hold on all six joints. "
        f"J6 static calibration feedforward multiplier: {j6_scale:.0%}.",
        flush=True,
    )
    while True:
        start = time.monotonic()
        elapsed = start - started_at
        check_status(arm, expected_motion_mode=mit_mode if elapsed >= _MIT_MODE_CONFIRM_S else None)
        check_enabled(arm, "J6 static calibration")
        positions = arm.get_joint_positions()
        pose_elapsed = start - pose_started_at
        if pose_index < len(_J6_STATIC_TARGETS_DEG):
            target_deg = _J6_STATIC_TARGETS_DEG[pose_index]
            target_rad = math.radians(target_deg)
            # Negative elapsed time holds the measured pose through the firmware mode change.
            fraction = min(1.0, max(0.0, pose_elapsed / _J6_STATIC_MOVE_S))
            blend = fraction**2 * (3.0 - 2.0 * fraction)
            targets[5] = initial_j6 + (target_rad - initial_j6) * blend

        model_torque, feedforward = mit_torques(model, positions, width, limit, scale, j6_scale=j6_scale)
        send_mit(arm, targets, feedforward, 1.0)
        if pose_index < len(_J6_STATIC_TARGETS_DEG) and pose_elapsed >= sampling_at:
            motor = arm.get_joint_motor_state(6)
            velocity, current = float(motor.velocity), float(motor.current)
            if not math.isfinite(velocity) or not math.isfinite(current):
                raise RuntimeError("Invalid J6 motor feedback.")
            error_rad = target_rad - positions[5]
            pd_estimate = _MIT_HOLD_KP * error_rad - _MIT_HOLD_KD * velocity
            total_estimate = float(feedforward[5]) + pd_estimate
            stationary = abs(velocity) <= _J6_STATIC_MAX_SPEED_RAD_S
            writer.writerow(
                [
                    elapsed,
                    pose_index + 1,
                    target_deg,
                    *np.rad2deg(positions),
                    math.degrees(error_rad),
                    velocity,
                    current,
                    model_torque[5],
                    feedforward[5],
                    pd_estimate,
                    total_estimate,
                    int(stationary),
                ]
            )
            if stationary:
                summaries.append((math.degrees(positions[5]), pd_estimate, total_estimate, current))
            if pose_elapsed >= pose_duration:
                output.flush()
                if summaries:
                    angle, pd, total, amps = map(median, zip(*summaries, strict=True))
                    print(
                        f"J6 {target_deg:+d} deg ({len(summaries)} stationary samples): "
                        f"measured {angle:+.2f} deg, estimated PD {pd:+.3f} N m, "
                        f"estimated total {total:+.3f} N m, current {amps:+.3f} A.",
                        flush=True,
                    )
                else:
                    print(f"J6 {target_deg:+d} deg: no stationary samples.", flush=True)
                summaries.clear()
                pose_index += 1
                initial_j6 = target_rad
                pose_started_at = start
                if pose_index == len(_J6_STATIC_TARGETS_DEG):
                    print(
                        f"J6 calibration saved to {output.name}. MIT position hold remains active. "
                        "Support the arm before stopping with Ctrl+C.",
                        flush=True,
                    )
        time.sleep(max(0.0, _COMMAND_PERIOD_S - (time.monotonic() - start)))


def _confirm_start(port: str, *, calibrate_j6: bool = False) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("--execute requires an interactive terminal for the right-arrow confirmation.")
    confirmed = Event()

    def on_key(key: str) -> None:
        if key == "right":
            confirmed.set()

    listener = TerminalKeyListener(on_key)
    print(
        f"{port}: The arm may move or lose holding torque during a mode change. "
        "Place it where it can safely lose holding torque, clear its range of motion, "
        "and ensure the emergency stop is accessible.",
        flush=True,
    )
    try:
        listener.start()
        action = (
            "switch modes, home the arm, and run J6 calibration from -45 to +45 deg"
            if calibrate_j6
            else "switch modes, home the arm, and enter MIT"
        )
        print(f"Press the Right Arrow key to {action}. Ctrl+C cancels.", flush=True)
        confirmed.wait()
    finally:
        listener.stop()


def main() -> None:
    """Read one leader's joint commands or run the host-controlled MIT experiment."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--port", default="can_left_main")
    parser.add_argument("--tool", choices=get_args(PiperXTool), required=True)
    parser.add_argument("--gripper-width-m", type=float, default=0.0)
    parser.add_argument("--gravity-base-m-s2", type=float, nargs=3, default=(0.0, 0.0, -9.81))
    parser.add_argument("--max-torque-nm", type=float, default=8.0, help="Per-joint feedforward limit in N m")
    parser.add_argument("--torque-scale", type=float, default=1.0)
    parser.add_argument(
        "--j6-drag-torque-scale",
        type=float,
        help="Multiply J6 feedforward during the MIT handoff and zero-drag (default: 1.0)",
    )
    parser.add_argument(
        "--home-duration-s",
        type=float,
        default=_HOME_DURATION_S,
        help="Homing trajectory duration in seconds",
    )
    parser.add_argument(
        "--can-loopback", action="store_true", help="Expose transmitted CAN frames to local diagnostic tools"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Confirm CAN control, enable the arm, home, and enter MIT mode"
    )
    parser.add_argument(
        "--j6-static-csv",
        type=Path,
        help="Keep MIT position hold and record the J6 static calibration to a new CSV file",
    )
    parser.add_argument(
        "--j6-static-torque-scale",
        type=float,
        help="Multiply J6 feedforward only during static calibration (default: 1.0)",
    )
    args = parser.parse_args()
    if args.j6_static_csv is not None and not args.execute:
        parser.error("--j6-static-csv requires --execute")
    j6_drag_scale = args.j6_drag_torque_scale
    if j6_drag_scale is None:
        j6_drag_scale = 1.0
    else:
        if not args.execute or args.j6_static_csv is not None:
            parser.error("--j6-drag-torque-scale requires --execute without --j6-static-csv")
        if not 0.0 <= j6_drag_scale <= 1.0:
            parser.error("--j6-drag-torque-scale must be in [0, 1]")
    j6_static_scale = args.j6_static_torque_scale
    if j6_static_scale is None:
        j6_static_scale = 1.0
    else:
        if args.j6_static_csv is None:
            parser.error("--j6-static-torque-scale requires --j6-static-csv")
        if not 0.0 <= j6_static_scale <= 1.0:
            parser.error("--j6-static-torque-scale must be in [0, 1]")
    if not math.isfinite(args.max_torque_nm) or args.max_torque_nm <= 0.0:
        parser.error("--max-torque-nm must be finite and positive")
    if not 0.0 < args.torque_scale <= 1.0:
        parser.error("--torque-scale must be in (0, 1]")
    if not math.isfinite(args.home_duration_s) or args.home_duration_s <= 0.0:
        parser.error("--home-duration-s must be finite and positive")
    config = PiperArmConfig(port=args.port)
    try:
        if args.execute:
            _confirm_start(args.port, calibrate_j6=args.j6_static_csv is not None)
        with ExitStack() as resources:
            calibration_output = None
            if args.j6_static_csv is not None:
                calibration_output = resources.enter_context(args.j6_static_csv.open("x", newline=""))
            config = replace(config, firmware=firmware_profile(config))
            arm = PiperXLeaderArm(config, max_feedback_age=0.1)
            arm.connect(resources, init_gripper=False, local_loopback=args.can_loopback)
            if calibration_output is not None:
                lower, upper = arm.limits[5]
                if not all(lower <= math.radians(angle) <= upper for angle in _J6_STATIC_TARGETS_DEG):
                    raise RuntimeError("J6 limits do not permit the -45 to +45 deg calibration.")
            if args.execute:
                model = PiperXDynamics(args.tool, args.gravity_base_m_s2)
                model.torques([0.0] * 6, gripper_width_m=args.gripper_width_m)
                try:
                    switched = ensure_can_control(arm, operator_confirmed=True)
                except RuntimeError as exc:
                    raise RuntimeError(f"{args.port}: CAN control setup failed: {exc}") from exc
                if switched:
                    print(f"{args.port}: CAN_CTRL confirmed", flush=True)
            deadline = time.monotonic() + 2.0
            while True:
                try:
                    if args.execute:
                        check_status(arm)
                        joints = arm.get_joint_positions()
                        enabled = arm.get_joint_enable_states()
                    else:
                        joints = arm.get_leader_joint_positions()
                        enabled = None
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
            angles = np.round(np.rad2deg(joints), 2).tolist()
            if enabled is None:
                print(f"Leader joint commands J1–J6 (deg): {angles}", flush=True)
                return
            print(f"Measured J1–J6 (deg): {angles}", flush=True)
            print(f"Motor enable J1–J6: {enabled}", flush=True)
            resources.callback(disable, arm)
            arm.enable(timeout_s=3.0)
            print(f"Homing via move_js to J1–J6 = 0° over {args.home_duration_s:g} s at 100 Hz.", flush=True)
            _home(arm, args.home_duration_s)
            if calibration_output is None:
                drag_ready = Event()
                return_requested = Event()

                def on_key(key: str) -> None:
                    if key == "left" and drag_ready.is_set():
                        return_requested.set()

                listener = TerminalKeyListener(on_key)
                resources.callback(listener.stop)
                listener.start()
                positions = _drag(
                    arm,
                    model,
                    args.gripper_width_m,
                    args.max_torque_nm,
                    args.torque_scale,
                    j6_scale=j6_drag_scale,
                    drag_ready=drag_ready,
                    return_requested=return_requested,
                )
                _return_to_js(arm, positions, args.home_duration_s)
            else:
                _calibrate_j6(
                    arm,
                    model,
                    args.gripper_width_m,
                    args.max_torque_nm,
                    args.torque_scale,
                    calibration_output,
                    j6_scale=j6_static_scale,
                )
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
    except (RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
