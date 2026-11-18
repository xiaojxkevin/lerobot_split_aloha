# Split-ALOHA station parameters

`controller_limits.json` records the left Piper-X controller limits as the
target for both sides. All six joints use the same speed and acceleration
limits. At startup the follower reads both controllers, applies any differing
settings, and checks the SDK's acknowledgment and readback before enabling.
The file's `firmware_profile` is used unless the per-arm profile is set.
To use another station configuration, pass `--robot.controller_limits_path`.

The two `calibration/follower_*.json` files were copied from the station's
Hugging Face `lerobot/calibration/robots/piperx_follower` cache. The old joint
ranges use millidegrees. To clip outgoing joint commands using these
calibration ranges and the selected pyAgxArm driver's limits, pass
`--robot.follower_calibration_dir=assets/split-aloha/calibration`. This narrows
each accepted joint range to the intersection. Gripper travel remains configured
in metres via `gripper_max_width_m` per arm.
