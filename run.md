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
  --agent.run-name baseline_arm_ee
```

恢复训练：

```bash
uv run go2arm-train Go2Arm-Flat \
  --agent.resume True \
  --agent.load-run ".*" \
  --agent.load-checkpoint "model_.*.pt"
```
继续上一个checkpoint训练：

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 4096 \
  --agent.resume True \
  --agent.load-run 2026-05-18_23-35-48_baseline_arm_ee \
  --agent.load-checkpoint model_1000.pt \
  --agent.max-iterations 2000 \
  --agent.save-interval 200 \
  --agent.run-name baseline_arm_ee_resume_1000
```

## Play

播放最新策略：

```bash
uv run go2arm-play Go2Arm-Flat
```

播放指定 checkpoint：

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/2026-05-18_23-35-48_baseline_arm_ee/model_1000.pt
```

录制视频：

```bash
uv run go2arm-play Go2Arm-Flat \
  --checkpoint-file logs/rsl_rl/go2arm_flat/2026-05-18_14-03-49/model_199.pt \
  --video True \
  --video-length 500
```

## TensorBoard

```bash
uv run tensorboard --logdir logs/rsl_rl/go2arm_flat
```
