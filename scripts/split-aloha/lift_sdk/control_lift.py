#!/usr/bin/env python3
"""Interactively control the Split ALOHA lift through its ROS 2 service.

Start ``service_850pro`` first (see ``/home/agilex/Desktop/data_collection.md``),
then run this program from a terminal with the ROS environments sourced.
"""


import argparse
import math
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass


VELOCITY_MODE = -4
POSITION_MODE = 0
STOP_MODE = -2


@dataclass(frozen=True)
class LiftControlConfig:
    speed_mm_s: int
    command_rate_hz: float
    min_height_mm: float
    max_height_mm: float
    feedback_timeout_s: float
    ready_timeout_s: float

    def validate(self) -> None:
        if self.speed_mm_s <= 0:
            raise ValueError("speed-mm-s must be positive")
        if not math.isfinite(self.command_rate_hz) or self.command_rate_hz <= 0:
            raise ValueError("command-rate-hz must be positive and finite")
        if not math.isfinite(self.min_height_mm) or self.min_height_mm < 0:
            raise ValueError("min-height-mm must be non-negative and finite")
        if not math.isfinite(self.max_height_mm) or self.max_height_mm <= self.min_height_mm:
            raise ValueError("max-height-mm must be greater than min-height-mm")
        if not math.isfinite(self.feedback_timeout_s) or self.feedback_timeout_s <= 0:
            raise ValueError("feedback-timeout-s must be positive and finite")
        if not math.isfinite(self.ready_timeout_s) or self.ready_timeout_s <= 0:
            raise ValueError("ready-timeout-s must be positive and finite")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manually control the Split ALOHA lift.")
    parser.add_argument("--speed-mm-s", type=int, default=50, help="Up/down speed (default: 50)")
    parser.add_argument("--command-rate-hz", type=float, default=10.0, help="Velocity heartbeat rate")
    parser.add_argument("--min-height-mm", type=float, default=0.0)
    parser.add_argument("--max-height-mm", type=float, default=780.0)
    parser.add_argument("--feedback-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--ready-timeout-s",
        type=float,
        default=90.0,
        help="Maximum time to wait for driver initialization and first lift feedback (default: 90)",
    )
    parser.add_argument("--state-topic", default="/LiftMotorStatePub")
    parser.add_argument("--service-name", default="/LiftingMotorService")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = LiftControlConfig(
        speed_mm_s=args.speed_mm_s,
        command_rate_hz=args.command_rate_hz,
        min_height_mm=args.min_height_mm,
        max_height_mm=args.max_height_mm,
        feedback_timeout_s=args.feedback_timeout_s,
        ready_timeout_s=args.ready_timeout_s,
    )
    config.validate()
    if not sys.stdin.isatty():
        raise RuntimeError("This controller must be run from an interactive terminal.")

    try:
        import rclpy
        from lifting_msg_pkg.msg import LiftMotorMsg
        from lifting_msg_pkg.srv import LiftMotorSrv
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError(
            "ROS lift packages are unavailable. Run: source /opt/ros/humble/setup.bash && "
            "source ~/agilex_ws/install/setup.bash"
        ) from exc

    class LiftController(Node):
        def __init__(self) -> None:
            super().__init__("robot_lab_manual_lift_control")
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
            )
            self.height_mm = float("nan")
            self.up_limit = False
            self.down_limit = False
            self.last_feedback_at = float("-inf")
            self.pending = None
            self.create_subscription(LiftMotorMsg, args.state_topic, self._on_state, qos)
            self.client = self.create_client(LiftMotorSrv, args.service_name)

        def _on_state(self, message: LiftMotorMsg) -> None:
            self.height_mm = float(message.back_height)
            self.up_limit = bool(message.up_limit)
            self.down_limit = bool(message.down_limit)
            self.last_feedback_at = time.monotonic()

        def command(self, mode: int, value: int) -> bool:
            if self.pending is not None:
                if not self.pending.done():
                    return False
                self.pending.result()
                self.pending = None
            request = LiftMotorSrv.Request()
            request.mode = mode
            request.val = value
            self.pending = self.client.call_async(request)
            return True

        def stop(self) -> None:
            # A stop is intentionally sent even if a previous velocity request is still pending.
            request = LiftMotorSrv.Request()
            request.mode = STOP_MODE
            request.val = 0
            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)

    rclpy.init()
    controller = LiftController()
    terminal_settings = termios.tcgetattr(sys.stdin.fileno())
    direction = 0
    target_height_mm: int | None = None
    next_command_at = time.monotonic()
    last_report_at = float("-inf")

    try:
        print(
            "Waiting for lift initialization, feedback, and service "
            f"(up to {config.ready_timeout_s:g}s)...",
            flush=True,
        )
        ready_deadline = time.monotonic() + config.ready_timeout_s
        while time.monotonic() < ready_deadline:
            rclpy.spin_once(controller, timeout_sec=0.05)
            if math.isfinite(controller.height_mm) and controller.client.service_is_ready():
                break
        else:
            raise TimeoutError(
                f"No ready feedback on {args.state_topic} or service {args.service_name}. "
                "Check that service_850pro finished initialization."
            )

        tty.setcbreak(sys.stdin.fileno())
        print(
            "Ready. P=up, L=down, G=go to height, Space=stop, Q=quit. "
            "Keep the lift area clear.",
            flush=True,
        )
        while True:
            now = time.monotonic()
            rclpy.spin_once(controller, timeout_sec=0.02)
            if now - controller.last_feedback_at > config.feedback_timeout_s:
                raise RuntimeError("Lift feedback became stale; stopping.")

            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                key = sys.stdin.read(1).lower()
                if key == "p":
                    direction = 1
                    target_height_mm = None
                elif key == "l":
                    direction = -1
                    target_height_mm = None
                elif key == "g":
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings)
                    try:
                        requested_height = float(
                            input(
                                f"\nTarget height in mm "
                                f"({config.min_height_mm:g}–{config.max_height_mm:g}): "
                            )
                        )
                    except ValueError:
                        print("Invalid height; no command sent.", flush=True)
                    else:
                        if not math.isfinite(requested_height):
                            print("Height must be finite; no command sent.", flush=True)
                        elif not config.min_height_mm <= requested_height <= config.max_height_mm:
                            print("Height is outside the configured travel range; no command sent.", flush=True)
                        elif controller.command(POSITION_MODE, round(requested_height)):
                            direction = 0
                            target_height_mm = round(requested_height)
                            print(f"Moving to {target_height_mm} mm. Press Space to stop.", flush=True)
                        else:
                            print("Previous lift service request is still pending; try again shortly.", flush=True)
                    tty.setcbreak(sys.stdin.fileno())
                elif key in {" ", "q", "\x03"}:
                    direction = 0
                    target_height_mm = None
                    controller.command(STOP_MODE, 0)
                    if key in {"q", "\x03"}:
                        break

            velocity = direction * config.speed_mm_s
            if velocity > 0 and (controller.up_limit or controller.height_mm >= config.max_height_mm):
                print("Upper limit reached; stopping.", flush=True)
                direction = 0
                velocity = 0
            elif velocity < 0 and (controller.down_limit or controller.height_mm <= config.min_height_mm):
                print("Lower limit reached; stopping.", flush=True)
                direction = 0
                velocity = 0

            if velocity != 0 and now >= next_command_at:
                controller.command(VELOCITY_MODE, velocity)
                next_command_at = now + 1.0 / config.command_rate_hz
            if target_height_mm is not None and abs(controller.height_mm - target_height_mm) <= 2.0:
                print(f"Target reached: {controller.height_mm:.1f} mm", flush=True)
                target_height_mm = None
            if now - last_report_at >= 0.5:
                target_text = "-" if target_height_mm is None else f"{target_height_mm} mm"
                print(
                    f"height={controller.height_mm:6.1f} mm  command={velocity:+4d} mm/s  "
                    f"target={target_text}  limits=({'UP' if controller.up_limit else '-'},"
                    f"{'DOWN' if controller.down_limit else '-'})",
                    flush=True,
                )
                last_report_at = now
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, terminal_settings)
        try:
            controller.stop()
        finally:
            controller.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
