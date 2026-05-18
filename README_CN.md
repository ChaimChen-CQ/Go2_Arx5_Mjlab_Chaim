# Go2 + ARX5 MJLab 训练项目

[English README](README.md)

这个仓库包含一个基于 MJLab / MuJoCo 的强化学习训练项目，机器人模型是
Unitree Go2 四足机器人加 ARX5 机械臂。

当前注册的任务名是：

```text
Go2Arm-Flat
```

当前训练目标是在平地上进行 Go2 底盘速度跟踪，同时给 ARX5 机械臂末端一个随机
位姿目标。机械臂末端目标格式是：

```text
[x, y, z, qw, qx, qy, qz]
```

其中 `x, y, z` 是机器人 base 坐标系下的目标位置，`qw, qx, qy, qz` 是目标四元数。

## 功能

- Go2 + ARX5 组合 MJCF 模型。
- MJLab manager-based RL 环境配置。
- RSL-RL PPO 训练。
- Go2 底盘平地速度命令。
- ARX5 机械臂随机末端位姿命令。
- 末端位置和姿态跟踪奖励。
- TensorBoard 记录末端位置、目标位置、四元数、roll / pitch / yaw 和误差。
- 保存 checkpoint 时自动导出 `policy.onnx`。

## 仓库结构

```text
assets/go2arm/
  models/                 MJCF 模型
  meshes/                 Go2 和 ARX5 mesh

src/go2arm_mjlab/
  assets/robots/go2arm/   机器人常量和 actuator 配置
  tasks/go2arm/           环境、MDP 项、奖励、metrics、commands
  scripts/                CLI 入口

scripts/                  简单脚本 wrapper
run.md                    常用训练、播放、TensorBoard 命令
TRAINING_NOTES.md         实验记录和修改日志
```

## 安装

本项目使用 `uv`。

```bash
cd /home/chaim/go2arm_mjlab
uv sync
```

查看可用任务：

```bash
uv run go2arm-list-envs
```

应该能看到：

```text
Go2Arm-Flat
```

## 训练

快速测试：

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 64 \
  --agent.max-iterations 200
```

正式训练：

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 2000 \
  --agent.save-interval 200 \
  --agent.run-name baseline_arm_ee_pose
```

训练日志会保存到：

```text
logs/rsl_rl/go2arm_flat/
```

checkpoint 文件形式：

```text
model_0.pt
model_200.pt
model_400.pt
...
```

每次保存 checkpoint 时，同目录下也会导出：

```text
policy.onnx
```

注意：`policy.onnx` 会被最新保存的策略覆盖，不是每个 checkpoint 单独保存一个 ONNX。

## 继续训练

从相同 observation / policy 结构的 checkpoint 继续训练：

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run <训练目录> \
  --agent.load-checkpoint <checkpoint.pt> \
  --agent.max-iterations 2000 \
  --agent.save-interval 200 \
  --agent.run-name baseline_arm_ee_pose_resume
```

注意：旧 observation 结构训练出来的 checkpoint 不能直接用于当前的末端位姿命令配置。

## 播放

播放最新策略：

```bash
uv run go2arm-play Go2Arm-Flat
```

播放指定 checkpoint：

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/<训练目录>/model_1999.pt
```

录制视频：

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/<训练目录>/model_1999.pt \
  --video True \
  --video-length 500
```

视频通常保存到：

```text
logs/rsl_rl/go2arm_flat/<训练目录>/videos/play/
```

## TensorBoard

```bash
uv run tensorboard --logdir logs/rsl_rl/go2arm_flat
```

常用末端指标包括：

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

## 说明

- 机器人仿真里有 20 个 joint，但 policy action 是 19 维。
- `joint8` 是夹爪 mimic joint，和 `joint7` 通过约束联动，所以只观测，不单独控制。
- `logs/`、checkpoint、ONNX、视频、虚拟环境和 zip 文件不会提交到 git。
- 实验过程、问题分析、修改和验证结果见 `TRAINING_NOTES.md`。
