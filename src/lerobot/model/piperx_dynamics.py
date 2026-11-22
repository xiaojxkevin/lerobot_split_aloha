"""Compute Piper-X model torques from joint state in SDK units."""

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from xml.etree import ElementTree as ET

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lerobot.utils.import_utils import _pinocchio_available, require_package

if TYPE_CHECKING or _pinocchio_available:
    import pinocchio as pin


PiperXTool = Literal["teach-pendant", "gripper", "gripper-d435"]
_URDFS = {
    "teach-pendant": "piper_x_with_teach_pendant.urdf",
    "gripper": "piper_x_with_gripper.urdf",
    "gripper-d435": "piper_x_with_gripper_d435.urdf",
}
_CUROBO_URDF_DIR = Path("content/assets/robot/piper_x_description/urdf")
_ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
_FINGER_JOINTS = ("gripper_joint1", "gripper_joint2")


def _read_urdf(tool: PiperXTool) -> str:
    relative_path = _CUROBO_URDF_DIR / _URDFS[tool]
    checkout = Path(__file__).resolve().parents[3] / "third_party" / "curobo" / "curobo"
    source = checkout / relative_path
    if source.is_file():
        return source.read_text(encoding="utf-8")
    try:
        resource = files("curobo").joinpath(*relative_path.parts)
    except ModuleNotFoundError as exc:
        if exc.name != "curobo":
            raise
    else:
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"cuRobo is missing {relative_path}. Use the project's third_party/curobo checkout "
        "or install the cuRobo fork with the Piper-X dynamics assets."
    )


def _finite_vector(values: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain {size} finite values.")
    return vector


class PiperXDynamics:
    """Evaluate one Piper-X arm with a cuRobo payload model."""

    def __init__(self, tool: PiperXTool, gravity_base_m_s2: ArrayLike = (0.0, 0.0, -9.81)) -> None:
        """Load a tool variant and set the gravity vector in base coordinates."""
        require_package("pin", "kinematics", import_name="pinocchio")
        # Model-only parsing avoids ROS package resolution and mesh loading.
        xml = _read_urdf(tool)
        self._model = pin.buildModelFromXML(xml)
        self._model.gravity.linear = _finite_vector(gravity_base_m_s2, 3, "Base gravity")
        self._data = self._model.createData()
        self._arm_q = [self._joint(name).idx_q for name in _ARM_JOINTS]
        self._arm_v = [self._joint(name).idx_v for name in _ARM_JOINTS]
        self._width_q: int | None = None
        self._finger_q: list[int] = []
        self._finger_multipliers: list[float] = []
        self._finger_offsets: list[float] = []
        self._min_width = self._max_width = 0.0
        if tool != "teach-pendant":
            self._width_q = self._joint("gripper").idx_q
            self._finger_q = [self._joint(name).idx_q for name in _FINGER_JOINTS]
            robot = ET.fromstring(xml)
            for name in _FINGER_JOINTS:
                mimic = robot.find(f"joint[@name='{name}']/mimic")
                if mimic is None or mimic.get("joint") != "gripper":
                    raise ValueError(f"Piper-X finger {name!r} must mimic the gripper joint.")
                self._finger_multipliers.append(float(mimic.get("multiplier", "1")))
                self._finger_offsets.append(float(mimic.get("offset", "0")))
            self._min_width = float(self._model.lowerPositionLimit[self._width_q])
            self._max_width = float(self._model.upperPositionLimit[self._width_q])
        self._zero = np.zeros(self._model.nv)

    def _joint(self, name: str) -> "pin.JointModel":
        model = self._model
        if not model.existJointName(name):
            raise ValueError(f"Piper-X model is missing joint {name!r}.")
        joint = model.joints[model.getJointId(name)]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(f"Piper-X joint {name!r} must have one position and velocity coordinate.")
        return joint

    def torques(
        self,
        joint_rad: ArrayLike,
        *,
        gripper_width_m: float = 0.0,
        joint_velocity_rad_s: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Return the six arm torques in N·m, ordered from J1 to J6.

        With positions alone, return gravity compensation. With joint velocities,
        include the velocity-dependent term using inverse dynamics with zero
        acceleration. The gripper remains at the specified width in both cases.
        """
        if not self._min_width <= gripper_width_m <= self._max_width:
            raise ValueError(f"Gripper width must be in [{self._min_width}, {self._max_width}] m.")
        q = np.zeros(self._model.nq)
        q[self._arm_q] = _finite_vector(joint_rad, 6, "Joint angles")
        if self._width_q is not None:
            q[self._width_q] = gripper_width_m
            # The default parser exposes both finger coordinates independently.
            q[self._finger_q] = np.multiply(self._finger_multipliers, gripper_width_m) + self._finger_offsets
        if joint_velocity_rad_s is None:
            tau = pin.computeGeneralizedGravity(self._model, self._data, q)
        else:
            v = np.zeros(self._model.nv)
            v[self._arm_v] = _finite_vector(joint_velocity_rad_s, 6, "Joint velocities")
            tau = pin.rnea(self._model, self._data, q, v, self._zero)
        return np.asarray(tau, dtype=np.float64)[self._arm_v]


def main() -> None:
    """Print the selected model's torques for a pose without opening CAN."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--tool", choices=_URDFS, required=True)
    parser.add_argument("--joint-rad", type=float, nargs=6, required=True)
    parser.add_argument("--joint-velocity-rad-s", type=float, nargs=6)
    parser.add_argument("--gripper-width-m", type=float, default=0.0)
    parser.add_argument("--gravity-base-m-s2", type=float, nargs=3, default=(0.0, 0.0, -9.81))
    args = parser.parse_args()
    dynamics = PiperXDynamics(args.tool, args.gravity_base_m_s2)
    torque = dynamics.torques(
        args.joint_rad,
        gripper_width_m=args.gripper_width_m,
        joint_velocity_rad_s=args.joint_velocity_rad_s,
    )
    print(json.dumps({"tool": args.tool, "joint_torque_nm": torque.tolist()}, indent=2))


if __name__ == "__main__":
    main()
