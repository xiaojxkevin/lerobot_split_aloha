# LeRobot for AgileX Split-aloha (piperx + ranger mini 3.0)

## Split-ALOHA client setup

From the repository root, install the Python 3.12 environment and client dependencies.

```bash
uv sync --locked --python 3.12 \
  --extra dataset --extra intelrealsense --extra pyserial-dep

uv pip install --python .venv/bin/python \
  'git+https://github.com/agilexrobotics/pyAgxArm.git' \
  'websockets==15.0.1' 'msgpack>=1.1,<2'

uv pip install --python .venv/bin/python --no-deps -e ./openpi-client
```

The bundled OpenPI client's package metadata pins NumPy 1.x. Its WebSocket codec
works with the NumPy 2.x version locked by this repo. Use `uv run --no-sync`
for the commands below. If you run `uv sync` again, repeat the extra installs
and any PyTorch override.

For an RTX 4060 host with a working NVIDIA 570-series driver, install the
CUDA 12.8 GPU wheels after the locked sync. On a host where the default locked
CUDA build can allocate a CUDA tensor, keep that build.

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
`piper_sdk/init_can.sh` under that directory sets up the two follower interfaces.
Leader recording also needs separate Linux CAN interfaces for the two leaders.

`bi_piperx_follower` provides absolute left/right joint targets in radians, with
six joints and one gripper field per side. The gripper fields retain the D4RT
training encoding `width_m * 1000 * pi / 180` and the SDK receives metres.
Override `--robot.left_arm_config.firmware` and
`--robot.right_arm_config.firmware` to match the controller's
[pyAgxArm firmware profile](https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/piper/firmware_reference.md).

For a hardware test, replay the same small joint trajectory using
`lerobot-replay` with each command mode. The default is
`--robot.joint_command=move_j`. To select `move_js`, set
`--robot.joint_command=move_js`. The follower clips joint and gripper targets
to their configured travel. Compare the two runs at the same rate and initial pose.
Rollout can refresh the last target in `move_j` mode during a brief RTC gap.
For `move_js`, an exhausted RTC queue ends the rollout.

For leader capture use `lerobot-record --robot.type=bi_piperx_follower
--teleop.type=bi_piperx_leader` with the usual dataset and camera options.
The default leader ports are `can_left_main` and `can_right_main`.
Configure the two controllers in leader mode on their own buses before recording.
The adapter reads the SDK's leader joint angles and gripper width command.

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

## Robots & Control

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot provides a unified `Robot` class interface that decouples control logic from hardware specifics. It supports a wide range of robots and teleoperation devices.

```python
from lerobot.robots.myrobot import MyRobot

# Connect to a robot
robot = MyRobot(config=...)
robot.connect()

# Read observation and send action
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**Supported Hardware:** SO101, OpenARM, LeKiwi, Unitree G1, reBot B601, Koch, HopeJR, OMX, EarthRover, Reachy2, Gamepads, Keyboards, Phones and more.

While these devices are natively integrated into the LeRobot codebase, the library is designed to be extensible. You can easily implement the Robot interface to utilize LeRobot's data collection, training, and visualization tools for your own custom robot.

For detailed hardware setup guides, see the [Hardware Documentation](https://huggingface.co/docs/lerobot/integrate_hardware).

## LeRobot Dataset

To solve the data fragmentation problem in robotics, we utilize the **LeRobotDataset** format.

- **Structure:** Synchronized MP4 videos (or images) for vision and Parquet files for state/action data.
- **HF Hub Integration:** Explore thousands of robotics datasets on the [Hugging Face Hub](https://huggingface.co/lerobot).
- **Tools:** Seamlessly delete episodes, split by indices/fractions, add/remove features, and merge multiple datasets.

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset from the Hub
dataset = LeRobotDataset("lerobot/aloha_mobile_cabinet")

# Access data (automatically handles video decoding)
episode_index=0
print(f"{dataset[episode_index]['action'].shape=}\n")
```

Learn more about it in the [LeRobotDataset Documentation](https://huggingface.co/docs/lerobot/lerobot-dataset-v3).

## SoTA Models

LeRobot implements state-of-the-art policies in pure PyTorch, covering Imitation Learning, Reinforcement Learning, Vision-Language-Action (VLA) models, World Models, and Reward Models, with more coming soon. It also provides you with the tools to instrument and inspect your training process.

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

Training a policy is as simple as running a script configuration:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet \
  --policy.push_to_hub=false
```

| Category                   | Models                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Imitation Learning**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md), [Multitask DiT Policy](./docs/source/policy_multi_task_dit_README.md)                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Reinforcement Learning** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL (coming soon)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **VLAs Models**            | [Pi0](./docs/source/pi0.mdx), [Pi0Fast](./docs/source/pi0fast.mdx), [Pi0.5](./docs/source/pi05.mdx), [OpenGalaxea G0.5](./docs/source/g05.mdx), [FineART-VLA](./docs/source/fineart_vla.mdx), [GR00T N1.7](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx), [EO-1](./docs/source/eo1.mdx), [MolmoAct2](./docs/source/molmoact2.mdx), [LingBot-VLA 2.0](./docs/source/lingbot_vla_v2.mdx), [WALL-OSS](./docs/source/walloss.mdx), [Wall-OSS-0.5](./docs/source/wall_oss_05.mdx), [EVO1](./docs/source/evo1.mdx), [DM05](./docs/source/dm05.mdx) |
| **World Models**           | [VLA-JEPA](./docs/source/vla_jepa.mdx), [LingBot-VA](./docs/source/lingbot_va.mdx), [FastWAM](./docs/source/fastwam.mdx), [Long-WAM](./docs/source/longwam.mdx), [LaWAM](./docs/source/lawam.mdx), [FLUX 3 Action](./docs/source/flux3.mdx), [Cosmos3](./docs/source/cosmos3.mdx), [Praxis-1](./docs/source/praxis.mdx)                                                                                                                                                                                                                                                                                                 |
| **Reward Models**          | [SARM](./docs/source/sarm.mdx), [TOPReward](./docs/source/topreward.mdx), [Robometer](./docs/source/robometer.mdx)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

Similarly to the hardware, you can easily implement your own policy & leverage LeRobot's data collection, training, and visualization tools, and share your model to the HF Hub.

For detailed policy setup guides, see the [Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies). For GPU/RAM requirements and expected training time per policy, see the [Compute Hardware Guide](https://huggingface.co/docs/lerobot/hardware_guide).

## Inference & Evaluation

Evaluate your policies in simulation with `lerobot-eval`. LeRobot supports standard benchmarks like **LIBERO**, **MetaWorld** and more to come. To run a trained policy on a real robot, use `lerobot-rollout` (see the [Policy Deployment Documentation](https://huggingface.co/docs/lerobot/inference)).

```bash
# Evaluate a policy on the LIBERO benchmark
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

Learn how to implement your own simulation environment or benchmark and distribute it from the HF Hub by following the [EnvHub Documentation](https://huggingface.co/docs/lerobot/envhub).

### Third-Party Hardware

Beyond the natively supported hardware, the community maintains a growing ecosystem of plugins for other robots, teleoperators, cameras, and sensors - UFACTORY xArm, Universal Robots UR5e, Franka, AgileX Piper, Trossen WidowX, ARX5, I2RT YAM, GELLO, SpaceMouse, Meta Quest, ROS 2 bridges, tactile and depth cameras, and more.

Plugins are auto-discovered by package name: LeRobot imports any installed package prefixed with `lerobot_robot_`, `lerobot_teleoperator_`, or `lerobot_camera_`. Install one and use the `type` it registers straight from the CLI:

```bash
pip install lerobot_robot_<name> lerobot_teleoperator_<name>

lerobot-record \
  --robot.type=<robot_name> \
  --teleop.type=<teleoperator_name> \
  --dataset.repo_id=${HF_USER}/my-dataset
```

Browse the full list in the [Third-Party Robots & Teleoperators](https://huggingface.co/docs/lerobot/main/third_party_robots) and [Third-Party Cameras & Sensors](https://huggingface.co/docs/lerobot/main/third_party_sensors) documentation.

## Resources

- **[Documentation](https://huggingface.co/docs/lerobot/index):** The complete guide to tutorials & API.
- **[Chinese Tutorials: LeRobot+SO-ARM101中文教程-同济子豪兄](https://zihao-ai.feishu.cn/wiki/space/7589642043471924447)** Detailed doc for assembling, teleoperate, dataset, train, deploy. Verified by Seed Studio and 5 global hackathon players.
- **[Discord](https://discord.gg/s3KuuzsPFb):** Join the `LeRobot` server to discuss with the community.
- **[X](https://x.com/LeRobotHF):** Follow us on X to stay up-to-date with the latest developments.
- **[Robot Learning Tutorial](https://huggingface.co/spaces/lerobot/robot-learning-tutorial):** A free, hands-on course to learn robot learning using LeRobot.
- **[T-Shirt Folding Experiment](https://huggingface.co/spaces/lerobot/robot-folding):** An end-to-end demonstration of folding t-shirts with LeRobot.
- **[LeLab](https://github.com/huggingface/leLab):** A web interface for LeRobot — teleoperate, calibrate, record datasets, replay, and train your SO arm from the browser, no CLI required.

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
