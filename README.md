# Go2 + ARX5 MJLab Training

[中文说明](README_CN.md)

This repository contains a reinforcement learning training setup for a combined
Unitree Go2 quadruped and ARX5 robotic arm in MJLab / MuJoCo.

The main task is registered as:

```text
Go2Arm-Flat
```

The current training setup focuses on flat-terrain locomotion with an
end-effector pose command for the arm. The policy observes a random target pose
for the arm end effector:

```text
[x, y, z, qw, qx, qy, qz]
```

where `x, y, z` are target position values in the robot base frame and
`qw, qx, qy, qz` are the target quaternion.

## Features

- Combined Go2 + ARX5 MJCF asset.
- MJLab manager-based RL task configuration.
- RSL-RL PPO training.
- Flat-terrain velocity command for the Go2 base.
- Random end-effector pose command for the ARX5 arm.
- End-effector position and orientation tracking rewards.
- TensorBoard metrics for end-effector position, target position, quaternion,
  roll / pitch / yaw, and tracking errors.
- Automatic ONNX export when checkpoints are saved.

## Repository Layout

```text
assets/go2arm/
  models/                 MJCF models
  meshes/                 Go2 and ARX5 meshes

src/go2arm_mjlab/
  assets/robots/go2arm/   Robot constants and actuator settings
  tasks/go2arm/           Environment, MDP terms, rewards, metrics, commands
  scripts/                CLI entry points

scripts/                  Thin script wrappers
run.md                    Common training, play, and TensorBoard commands
TRAINING_NOTES.md         Experiment notes and change log
```

## Installation

This project uses `uv`.

```bash
cd /home/chaim/go2arm_mjlab
uv sync
```

List available tasks:

```bash
uv run go2arm-list-envs
```

You should see:

```text
Go2Arm-Flat
```

## Training

Quick smoke test:

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 64 \
  --agent.max-iterations 200
```

Main training run:

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 2000 \
  --agent.save-interval 200 \
  --agent.run-name baseline_arm_ee_pose
```

Training logs are written to:

```text
logs/rsl_rl/go2arm_flat/
```

Checkpoint files are saved as:

```text
model_0.pt
model_200.pt
model_400.pt
...
```

The runner also exports:

```text
policy.onnx
```

in the same run directory whenever a checkpoint is saved. The ONNX file is
overwritten with the latest saved policy.

## Resume Training

Resume from a checkpoint with the same observation / policy structure:

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run <run-directory> \
  --agent.load-checkpoint <checkpoint.pt> \
  --agent.max-iterations 2000 \
  --agent.save-interval 200 \
  --agent.run-name baseline_arm_ee_pose_resume
```

Important: old checkpoints from earlier observation layouts are not compatible
with the current end-effector pose command configuration.

## Play

Play the latest policy:

```bash
uv run go2arm-play Go2Arm-Flat
```

Play a specific checkpoint:

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/<run-directory>/model_1999.pt
```

Record a video:

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/<run-directory>/model_1999.pt \
  --video True \
  --video-length 500
```

Videos are saved under the selected run directory, usually:

```text
logs/rsl_rl/go2arm_flat/<run-directory>/videos/play/
```

## TensorBoard

```bash
uv run tensorboard --logdir logs/rsl_rl/go2arm_flat
```

Useful end-effector metrics include:

```text
Episode_Metrics/ee_pos_x
Episode_Metrics/ee_pos_y
Episode_Metrics/ee_pos_z
Episode_Metrics/ee_target_x
Episode_Metrics/ee_target_y
Episode_Metrics/ee_target_z
Episode_Metrics/ee_pos_error
Episode_Metrics/ee_target_quat_w
Episode_Metrics/ee_target_quat_x
Episode_Metrics/ee_target_quat_y
Episode_Metrics/ee_target_quat_z
Episode_Metrics/ee_orientation_error
Episode_Metrics/ee_quat_w
Episode_Metrics/ee_quat_x
Episode_Metrics/ee_quat_y
Episode_Metrics/ee_quat_z
Episode_Metrics/ee_roll
Episode_Metrics/ee_pitch
Episode_Metrics/ee_yaw
```

## Notes

- The robot has 20 simulated joints but 19 policy actions.
- `joint8` is a gripper mimic joint coupled to `joint7`, so it is observed but
  not directly controlled by the policy.
- Logs, checkpoints, ONNX exports, videos, virtual environments, and zip
  archives are ignored by git.
- See `TRAINING_NOTES.md` for the experiment history, design decisions, and
  validation notes.
