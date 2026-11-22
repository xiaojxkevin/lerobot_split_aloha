"""Switch the Split-ALOHA Piper-X leaders to host CAN control."""

import argparse
import sys
from contextlib import ExitStack

from lerobot.piperx import PiperArmConfig
from lerobot.teleoperators.bi_piperx_leader.bi_piperx_leader import PiperXLeaderArm
from lerobot.teleoperators.bi_piperx_leader.can_control import ensure_can_control


def _ensure_can_control(port: str) -> None:
    with ExitStack() as resources:
        arm = PiperXLeaderArm(PiperArmConfig(port=port, firmware="v189"), max_feedback_age=0.25)
        arm.connect(resources, init_gripper=False)
        result = "CAN_CTRL confirmed" if ensure_can_control(arm) else "already in CAN_CTRL"
        print(f"{port}: {result}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ports", nargs="+", choices=("can_left_main", "can_right_main"))
    args = parser.parse_args()
    failed = False
    for port in args.ports:
        try:
            _ensure_can_control(port)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            print(f"{port}: CAN control setup failed: {exc}", file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
