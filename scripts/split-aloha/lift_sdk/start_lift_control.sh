#!/usr/bin/env bash
# Start the lift ROS driver and open the interactive lift controller.

set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ros_setup="/opt/ros/humble/setup.bash"
agilex_setup="${AGILEX_WS_SETUP:-$HOME/agilex_ws/install/setup.bash}"
driver_log="${LIFT_DRIVER_LOG:-/tmp/robot-lab-lift-driver.log}"
driver_pid=""

cleanup() {
  if [[ -n "$driver_pid" ]] && kill -0 "$driver_pid" 2>/dev/null; then
    echo "Stopping lift driver (PID $driver_pid)..."
    kill -TERM "$driver_pid" 2>/dev/null || true
    wait "$driver_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -f "$ros_setup" ]]; then
  echo "ROS setup script not found: $ros_setup" >&2
  exit 1
fi
if [[ ! -f "$agilex_setup" ]]; then
  echo "AgileX workspace setup script not found: $agilex_setup" >&2
  echo "Set AGILEX_WS_SETUP to the correct setup.bash path if needed." >&2
  exit 1
fi

source "$ros_setup"
source "$agilex_setup"
set -u

if [[ ! -r /dev/ttyUSB0 || ! -r /dev/ttyUSB1 ]]; then
  echo "Warning: /dev/ttyUSB0 or /dev/ttyUSB1 is not readable by this user." >&2
  echo "Run the documented udev/permission setup before continuing." >&2
fi

echo "Starting lift driver; log: $driver_log"
ros2 run lifting_ctrl_ros2_pkg service_850pro >"$driver_log" 2>&1 &
driver_pid=$!

echo "Waiting for /LiftingMotorService..."
for _ in $(seq 1 100); do
  if ros2 service list 2>/dev/null | grep -qx '/LiftingMotorService'; then
    echo "Lift service is available; waiting for the driver's initialization and state feedback..."
    /usr/bin/python3 "$script_dir/control_lift.py" "$@"
    exit $?
  fi
  if ! kill -0 "$driver_pid" 2>/dev/null; then
    echo "Lift driver exited unexpectedly. Log follows:" >&2
    tail -n 100 "$driver_log" >&2 || true
    exit 1
  fi
  sleep 0.1
done

echo "Timed out waiting for /LiftingMotorService. Log follows:" >&2
tail -n 100 "$driver_log" >&2 || true
exit 1
