# LeRobot for AgileX Split-aloha (piperx + ranger mini 3.0)

## Split-ALOHA client setup

From the repository root, install the Python 3.12 environment and client dependencies.

```bash
uv sync --locked --python 3.12 \
  --extra dataset --extra intelrealsense --extra pyserial-dep

uv pip install --python .venv/bin/python --no-deps -e .

uv pip install --python .venv/bin/python \
  'git+https://github.com/agilexrobotics/pyAgxArm.git' \
  'websockets==15.0.1' 'msgpack>=1.1,<2'

uv pip install --python .venv/bin/python --no-deps -e ./openpi-client

# MUJOCO
uv pip install --python .venv/bin/python 'mujoco==3.8.1'

```

The bundled OpenPI client's package metadata pins NumPy 1.x. Its WebSocket codec
works with the NumPy 2.x version locked by this repo, so install it with
`--no-deps` as shown above.

For devices with a NVIDIA 570-series driver, install the CUDA 12.8 GPU wheels after the locked sync. On a host where the default locked CUDA build can allocate a CUDA tensor, keep that build.

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
uv pip install --python .venv/bin/python \
  --index https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
```

`git submodule update --init third_party/curobo` checks out the project's cuRobo fork.
Install with:

```bash
uv pip install --python .venv/bin/python \
  'https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl' \
  -e 'third_party/curobo[cu12]'
```

### Piper-X control and deployment

The CAN and lift utilities live under [scripts/split-aloha](scripts/split-aloha).
From the repository root, initialize the four Piper-X CAN interfaces at 1 Mbps
using Sealab's USB bus-info mapping. The script also sets both leaders to
host CAN control and verifies the reported mode. A mode change requires all
six motors on the affected arm to be disabled.

```bash
bash scripts/split-aloha/piperx_sdk/init_can.sh
```

| Interface | USB bus-info |
| --- | --- |
| `can_left_main` | `1-6.4.4:1.0` |
| `can_right_main` | `1-6.4.3:1.0` |
| `can_left_slave` | `1-1:1.0` |
| `can_right_slave` | `1-12:1.0` |

For firmware teaching, pass `--interfaces-only` to keep the current arm modes.
For policy-only deployment, pass `--followers-only`. The
`--left-usb` and `--right-usb` options override follower USB addresses.

#### Piper-X dynamics models for MIT experiments

LeRobot reads the `teach-pendant`, `gripper`, and `gripper-d435` URDFs from the
cuRobo checkout or the installed cuRobo fork. The [cuRobo Piper-X model reference](third_party/curobo/curobo/content/assets/robot/piper_x_description/README.md)
records the upstream revision, payload assumptions, and generator location.

The PyPI package is `pin` and the import name is `pinocchio`. The repository's
locked `kinematics` extra also supplies Pinocchio.

```bash
uv pip install --python .venv/bin/python pin
uv run --no-sync python -m lerobot.model.piperx_dynamics \
  --tool gripper-d435 --joint-rad 0 0.78 -1.05 0 0.52 0 --gripper-width-m 0.04
```

The command prints J1–J6 gravity torques in N·m without opening CAN. For a
tilted base, pass the gravity vector in base coordinates with
`--gravity-base-m-s2`. Passing `--joint-velocity-rad-s` adds the
velocity-dependent term with joint accelerations set to zero.

#### Teleoperation and policy deployment

`bi_piperx_follower` provides absolute left/right joint targets in radians, with
six joints and one gripper field per side. The gripper fields retain the D4RT
training encoding `width_m * 1000 * pi / 180` and the SDK receives metres.
At startup, the follower reads [controller_limits.json](assets/split-aloha/controller_limits.json).
It queries both controllers and writes changed joint and flange limits before
enabling either arm. The file also selects the default firmware profile (`v189`).
Set `--robot.controller_limits_path` to use a different station file. The per-arm
`firmware` options can override the file's profile.

The follower sends all joint targets with the SDK's `move_js` high-follow
command. It clips joint and gripper targets to their configured travel.
An exhausted RTC action queue ends the rollout.

For host MIT demonstrations, use `lerobot-teleoperate` with
`--teleop.mit_zero_drag=true`. Each episode moves all four arms to joint zero
with `move_js`, releases the leaders into MIT gravity compensation, and then
moves all four arms back to zero. The followers track the leaders' measured
angles with `move_js`. Homing interpolates from each arm's own measured pose.
The terminal accepts **n** to end and save an episode, **r** to discard and
repeat, and **q** to finish recording after the current episode returns to zero.

```bash
bash scripts/split-aloha/record_demonstration.sh --dataset.num_episodes=1
```

The script defaults to the camera serials and task from the Sealab launcher.
Set `FRONT_SERIAL`, `LEFT_SERIAL`, `RIGHT_SERIAL`, or `TASK` to override them.
The script records locally in LeRobot v3.0 with three RGB streams,
`observation.state` from the followers' measured joints and grippers, and
`action` from the targets sent to the followers after travel limits are applied.
For live teleoperation without recording, omit the `--dataset.*` options from
the `lerobot-teleoperate` invocation. MIT teleoperation reads the pendant width
from fresh gripper status. In `feedback` mode the leader gripper motor is made
passive at episode start so the operator can open and close the jaws by hand;
the follower mirrors the measured width and the six leader arm motors remain
enabled. For a joints-only bench test, use
`--teleop.mit_gripper_mode=closed` to send a zero-width follower gripper target.

For firmware leader capture use `lerobot-record --robot.type=bi_piperx_follower
--teleop.type=bi_piperx_leader` with the usual dataset and camera options.
The default leader ports are `can_left_main` and `can_right_main`.
Configure the two controllers in leader mode on their own buses before recording.
The adapter reads the SDK's leader joint angles and gripper width command.
The [station assets](assets/split-aloha/README.md#leader-calibration-records)
include the left and right leader calibration records from the Hugging Face
LeRobot cache.

The following rollout command **enables the followers and executes model targets**.
Set the endpoint and camera serials for your station. Start the matching
RTC-trained joint checkpoint with `serve_d4rt_vla.py` on the inference host.

```bash
D4RT_ENDPOINT=ws://your-policy-host:8888
FRONT_SERIAL=your-front-serial
LEFT_SERIAL=your-left-wrist-serial
RIGHT_SERIAL=your-right-wrist-serial

uv run --no-sync lerobot-rollout \
  --robot.type=bi_piperx_follower \
  --robot.cameras="{fixed_front: {type: intelrealsense, serial_number_or_name: '$FRONT_SERIAL', width: 640, height: 480, fps: 30}, left_arm: {type: intelrealsense, serial_number_or_name: '$LEFT_SERIAL', width: 640, height: 480, fps: 30}, right_arm: {type: intelrealsense, serial_number_or_name: '$RIGHT_SERIAL', width: 640, height: 480, fps: 30}}" \
  --policy.type=pi_ws --policy.endpoint="$D4RT_ENDPOINT" \
  --policy.rtc_training_max_delay=10 \
  --inference.type=rtc --inference.rtc.mode=trained \
  --inference.rtc.execution_horizon=16 \
  --fps=30 --task='Fold the towel.' --return_to_initial_position=false
```

PiWS maps the three observation keys to `cam_high`, `cam_left_wrist` and
`cam_right_wrist`. It checks the server metadata before opening the CAN ports.
Training-time RTC sends an absolute `action_prefix[d,14]` and `rtc_delay=d`;
set `rtc_training_max_delay` to a value supported by the checkpoint. For a
synchronous baseline with a non-RTC server, use `--inference.type=sync` and
`--policy.rtc_training_max_delay=0`.


## Citation

If you use LeRobot in your project, please cite the GitHub repository to acknowledge the ongoing development and contributors:

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Meftah, Khalil and Ellerbach, Maxime and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

If you are referencing our research or the academic paper, please also cite our ICLR publication:

<details>
<summary><b>ICLR 2026 Paper</b></summary>

```bibtex
@inproceedings{cadenelerobot,
  title={LeRobot: An Open-Source Library for End-to-End Robot Learning},
  author={Cadene, Remi and Alibert, Simon and Capuano, Francesco and Aractingi, Michel and Zouitine, Adil and Kooijmans, Pepijn and Choghari, Jade and Russi, Martino and Pascal, Caroline and Palma, Steven and Shukor, Mustafa and Moss, Jess and Soare, Alexander and Aubakirova, Dana and Lhoest, Quentin and Gallou\'edec, Quentin and Wolf, Thomas},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2602.22818}
}
```

</details>
