import json
import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_args

from .config_bi_piperx_follower import PiperFirmware


@dataclass(frozen=True)
class PiperControllerLimits:
    """Motion limits stored by the Piper-X controller, in SI units."""

    firmware_profile: PiperFirmware
    joint_max_speed_rad_s: float
    joint_max_acceleration_rad_s2: float
    flange_max_linear_velocity_m_s: float
    flange_max_angular_velocity_rad_s: float
    flange_max_linear_acceleration_m_s2: float
    flange_max_angular_acceleration_rad_s2: float

    @classmethod
    def load(cls, path: Path) -> "PiperControllerLimits":
        with path.open() as file:
            limits = cls(**json.load(file))
        if limits.firmware_profile not in get_args(PiperFirmware):
            raise ValueError(f"{path}: invalid Piper firmware profile.")
        for field in fields(limits):
            if field.name == "firmware_profile":
                continue
            value = getattr(limits, field.name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path}: {field.name} must be a finite positive number.")
        return limits

    @staticmethod
    def _sync_limit(
        label: str,
        read: Callable[[], tuple[float, ...] | None],
        write: Callable[[], bool],
        desired: tuple[float, ...],
        tolerance: float,
    ) -> None:
        actual = read()
        for attempt in range(3):
            if actual is not None and all(
                math.isclose(current, target, rel_tol=0, abs_tol=tolerance)
                for current, target in zip(actual, desired, strict=True)
            ):
                return
            if attempt < 2:
                # Only write after a successful read. A missing ACK does not establish
                # whether a setting changed, so always query again before retrying.
                if actual is not None:
                    write()
                actual = read()
        raise RuntimeError(f"{label}: expected {desired}, controller readback was {actual}.")

    def apply(self, driver: Any, port: str | None) -> None:
        """Synchronize and independently verify the controller limits before enabling."""
        for joint in range(1, 7):

            def read_speed(joint_index: int = joint) -> tuple[float] | None:
                feedback = driver.get_joint_angle_vel_limits(joint_index, min_interval=0)
                return (feedback.msg.max_joint_spd,) if feedback is not None else None

            def read_acceleration(joint_index: int = joint) -> tuple[float] | None:
                feedback = driver.get_joint_acc_limits(joint_index, min_interval=0)
                return (feedback.msg.max_joint_acc,) if feedback is not None else None

            self._sync_limit(
                f"{port} J{joint} maximum speed (rad/s)",
                read_speed,
                lambda index=joint: driver.set_joint_angle_vel_limits(
                    index, max_joint_spd=self.joint_max_speed_rad_s, timeout=2
                ),
                (self.joint_max_speed_rad_s,),
                0.005,
            )
            self._sync_limit(
                f"{port} J{joint} maximum acceleration (rad/s²)",
                read_acceleration,
                lambda index=joint: driver.set_joint_acc_limits(
                    index, max_joint_acc=self.joint_max_acceleration_rad_s2, timeout=2
                ),
                (self.joint_max_acceleration_rad_s2,),
                0.005,
            )

        flange_limits = {
            "max_linear_vel": self.flange_max_linear_velocity_m_s,
            "max_angular_vel": self.flange_max_angular_velocity_rad_s,
            "max_linear_acc": self.flange_max_linear_acceleration_m_s2,
            "max_angular_acc": self.flange_max_angular_acceleration_rad_s2,
        }

        def read_flange() -> tuple[float, ...] | None:
            feedback = driver.get_flange_vel_acc_limits(min_interval=0)
            if feedback is None:
                return None
            return tuple(getattr(feedback.msg, f"end_{name}") for name in flange_limits)

        self._sync_limit(
            f"{port} flange limits (m/s, rad/s, m/s², rad/s²)",
            read_flange,
            lambda: driver.set_flange_vel_acc_limits(timeout=2, **flange_limits),
            tuple(flange_limits.values()),
            0.0005,
        )
