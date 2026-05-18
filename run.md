# Go2Arm Run

任务名：

```bash
Go2Arm-Flat
```

## 环境

```bash
uv sync
uv run go2arm-list-envs
```

## Train

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

恢复训练（只适合同一 observation 维度的 checkpoint）：

```bash
uv run go2arm-train Go2Arm-Flat \
  --agent.resume True \
  --agent.load-run ".*" \
  --agent.load-checkpoint "model_.*.pt"
```
继续上一个 checkpoint 训练示例（旧 `baseline_arm_ee` 没有末端四元数，不能用于当前 `baseline_arm_ee_pose` 配置）：

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run <同配置训练目录> \
  --agent.load-checkpoint <checkpoint.pt> \
  --agent.max-iterations 2000 \
  --agent.save-interval 200 \
  --agent.run-name baseline_arm_ee_pose_resume
```

## Play

播放最新策略：

```bash
uv run go2arm-play Go2Arm-Flat
```

播放指定 checkpoint：

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/<baseline_arm_ee_pose训练目录>/model_1999.pt
```

录制视频：

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/<baseline_arm_ee_pose训练目录>/model_1999.pt \
  --video True \
  --video-length 500
```

## TensorBoard

```bash
uv run tensorboard --logdir logs/rsl_rl/go2arm_flat
```
