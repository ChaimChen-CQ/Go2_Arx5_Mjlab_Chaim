# Go2Arm Training Notes

This file records training issues, hypotheses, code changes, and validation
results for the Go2 + ARX5 mjlab project.

## 2026-05-18: Baseline Run Setup

### Problem

Needed a repeatable command set for training, playing checkpoints, resuming
training, recording video, and checking TensorBoard logs.

### Idea

Keep all common commands in `run.md` so experiments can be started without
remembering CLI flags.

### Changes

- Added task name `Go2Arm-Flat`.
- Added quick test training command.
- Added formal training command.
- Added resume training command.
- Added play, checkpoint play, video recording, and TensorBoard commands.

### Test Result

- `uv run go2arm-list-envs` listed `Go2Arm-Flat`.
- Training generated logs under `logs/rsl_rl/go2arm_flat`.
- Play command could load checkpoints from the generated log directory.

## 2026-05-18: Low / Strange Body Posture

### Problem

The policy learned an unnatural low posture. The robot body was close to the
ground and the arm appeared to be used as a passive counterweight.

### Idea

The baseline locomotion reward came from a legged-locomotion style setup. It
tracked base velocity, but did not strongly enforce base height, upright body
orientation, or arm posture. The policy could exploit this by lowering the body
and letting the arm drift.

### Changes

Files changed:

- `src/go2arm_mjlab/tasks/go2arm/go2arm_env_cfg.py`
- `src/go2arm_mjlab/tasks/go2arm/mdp/rewards.py`

Main changes:

- Increased `flat_orientation_l2` penalty from `-1.0` to `-3.0`.
- Added `base_height_l2` with target height `0.40`.
- Added `arm_posture_l2` to regularize arm joints toward the default pose.
- Reduced arm action scale from `0.20` to `0.05` during the first stability fix.
- Tightened bad-orientation termination from `70 deg` to `45 deg`.

### Test Result

- Ran a smoke training test:

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 2 \
  --agent.max-iterations 1 \
  --agent.run-name smoke_pose_fix
```

- `RewardManager` loaded the new terms:
  - `flat_orientation_l2`
  - `base_height_l2`
  - `arm_posture_l2`
- Smoke log was removed after validation.

## 2026-05-18: Training Length and Save Interval

### Problem

The default / documented training setup was longer than desired and checkpoint
intervals were too sparse for iterative debugging.

### Idea

Use 2000 iterations and save every 200 iterations for the main experiment.

### Changes

Files changed:

- `run.md`
- `src/go2arm_mjlab/tasks/go2arm/config/__init__.py`

Main changes:

- Set formal command to `--agent.max-iterations 2000`.
- Set formal command to `--agent.save-interval 200`.
- Set default runner config `max_iterations=2000`.
- Set default runner config `save_interval=200`.

### Test Result

- `uv run go2arm-list-envs` still registered `Go2Arm-Flat`.
- New training runs saved checkpoints at interval 200.

## 2026-05-18: Logs Not Appearing

### Problem

After starting a new training run, it looked like logs were not generated.

### Idea

The run did generate logs, but only `model_0.pt` appeared because training
stopped before iteration 200. With `save_interval=200`, the next checkpoint is
`model_200.pt`.

### Investigation

Observed log directories:

```text
logs/rsl_rl/go2arm_flat/2026-05-18_21-14-58_baseline
logs/rsl_rl/go2arm_flat/2026-05-18_22-03-57_baseline
```

The new run contained:

- `events.out.tfevents...`
- `model_0.pt`
- `policy.onnx`
- `params/agent.yaml`
- `params/env.yaml`

### Test Result

- Confirmed no active `go2arm-train` process remained.
- Confirmed `params/agent.yaml` had:
  - `max_iterations: 2000`
  - `save_interval: 200`

## 2026-05-18: Front Legs Fold and Arm Does Not Move

### Problem

After the stability fix, the two front legs folded into an unnatural posture.
The mechanical arm also did not move, and there was no random end-effector
motion.

### Idea

Two separate issues were present:

- Front legs needed an explicit posture regularizer.
- The arm had action range and posture constraints, but no task objective.

The arm should receive a random end-effector target command, the policy should
observe that command, and reward should track the end-effector site.

### Changes

Files changed:

- `src/go2arm_mjlab/tasks/go2arm/go2arm_env_cfg.py`
- `src/go2arm_mjlab/tasks/go2arm/mdp/rewards.py`
- `src/go2arm_mjlab/tasks/go2arm/mdp/velocity_command.py`
- `run.md`

Main changes:

- Added `FRONT_LEG_ACTION_ORDER`.
- Added `front_leg_posture_l2` with weight `-0.35`.
- Added `UniformEndEffectorCommand`.
- Added `UniformEndEffectorCommandCfg`.
- Added `arm_ee_position` command with random base-frame target ranges:
  - `x=(0.20, 0.45)`
  - `y=(-0.25, 0.25)`
  - `z=(0.15, 0.45)`
- Added `ee_position_commands` observation.
- Added `track_end_effector_position` reward.
- Added `track_ee_position` with weight `1.0`.
- Relaxed `arm_posture_l2` from `-1.0` to `-0.05`.
- Increased arm action scale from `0.05` to `0.15`.
- Updated formal run name to `baseline_arm_ee`.

### Test Result

Ran a deeper smoke test:

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 8 \
  --agent.max-iterations 3 \
  --agent.save-interval 2 \
  --agent.run-name smoke_deeper_arm_ee
```

Observed:

- `CommandManager` contained:
  - `base_velocity`
  - `arm_ee_position`
- Actor observation shape changed from `680` to `710`.
- Critic observation shape changed to `980`.
- `RewardManager` contained:
  - `front_leg_posture_l2`
  - `track_ee_position`
  - `arm_posture_l2`
- Smoke log was removed after validation.

Important compatibility note:

- Old checkpoints with actor observation shape `680` are not compatible with
  the new actor observation shape `710`.
- Training should restart from scratch for `baseline_arm_ee`.

## 2026-05-19: Continue Training From Checkpoint

### Problem

Needed to continue training from:

```text
logs/rsl_rl/go2arm_flat/2026-05-18_23-35-48_baseline_arm_ee/model_1000.pt
```

### Idea

Use `--agent.resume True`, specify the run directory, and specify the checkpoint
filename.

### Command

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

### Test Result

- Command recorded in `run.md`.
- Full continuation run was not validated in this note.

## 2026-05-19: Add End-Effector TensorBoard Metrics

### Problem

TensorBoard showed reward terms such as `track_ee_position`, but did not expose
raw end-effector position or orientation values. This made it hard to diagnose
whether the arm was actually moving toward the sampled target or whether the
wrist orientation was flipping.

### Idea

Add metrics only, not observations or rewards. This keeps the actor / critic
input dimensions unchanged and preserves checkpoint compatibility while adding
debug visibility in TensorBoard.

### Changes

Files changed:

- `src/go2arm_mjlab/tasks/go2arm/mdp/metrics.py`
- `src/go2arm_mjlab/tasks/go2arm/mdp/__init__.py`
- `src/go2arm_mjlab/tasks/go2arm/go2arm_env_cfg.py`

Main changes:

- Added world-frame end-effector position metrics:
  - `ee_pos_x`
  - `ee_pos_y`
  - `ee_pos_z`
- Added world-frame end-effector target metrics:
  - `ee_target_x`
  - `ee_target_y`
  - `ee_target_z`
- Added position tracking error metric:
  - `ee_pos_error`
- Added end-effector quaternion metrics:
  - `ee_quat_w`
  - `ee_quat_x`
  - `ee_quat_y`
  - `ee_quat_z`
- Added end-effector roll / pitch / yaw metrics:
  - `ee_roll`
  - `ee_pitch`
  - `ee_yaw`

### Test Result

Ran a smoke test:

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 2 \
  --agent.max-iterations 1 \
  --agent.run-name smoke_ee_metrics
```

Observed:

- `MetricsManager` contained 15 active terms.
- New active metrics included:
  - `ee_pos_x/y/z`
  - `ee_target_x/y/z`
  - `ee_pos_error`
  - `ee_quat_w/x/y/z`
  - `ee_roll/pitch/yaw`
- Actor observation shape stayed `710`.
- Critic observation shape stayed `980`.
- Smoke log was removed after validation.

Note:

- These metrics appear in TensorBoard under `Episode_Metrics/...`.
- Because metrics are episode-averaged, they may not show meaningful nonzero
  values until environments reset at least once.

## 2026-05-19: Add End-Effector Quaternion Command

### Problem

The arm target command only specified end-effector position. It did not specify
end-effector orientation, so the wrist / gripper could reach a point while
facing an arbitrary direction.

### Idea

Upgrade the end-effector command from position-only to full pose:

```text
[x, y, z, qw, qx, qy, qz]
```

The first three values are a target position in the base frame. The last four
values are a target quaternion in the base frame.

### Changes

Files changed:

- `src/go2arm_mjlab/tasks/go2arm/mdp/velocity_command.py`
- `src/go2arm_mjlab/tasks/go2arm/mdp/rewards.py`
- `src/go2arm_mjlab/tasks/go2arm/mdp/metrics.py`
- `src/go2arm_mjlab/tasks/go2arm/go2arm_env_cfg.py`
- `run.md`

Main changes:

- Replaced `UniformEndEffectorCommand` with `UniformEndEffectorPoseCommand`.
- Replaced `arm_ee_position` command with `arm_ee_pose`.
- Added sampled orientation ranges:
  - `roll=(-0.8, 0.8)`
  - `pitch=(-0.8, 0.8)`
  - `yaw=(-1.57, 1.57)`
- Changed command observation from `ee_position_commands` to
  `ee_pose_commands`.
- Added `track_end_effector_orientation` reward.
- Added `track_ee_orientation` reward term with weight `0.5`.
- Added target quaternion and orientation error metrics:
  - `ee_target_quat_w/x/y/z`
  - `ee_orientation_error`
- Updated `run.md` to use `baseline_arm_ee_pose` for new runs.

### Test Result

Ran a smoke test:

```bash
uv run go2arm-train Go2Arm-Flat \
  --env.scene.num-envs 2 \
  --agent.max-iterations 1 \
  --agent.run-name smoke_ee_pose_quat
```

Observed:

- `CommandManager` contained `arm_ee_pose`.
- Actor observation shape changed from `710` to `750`.
- Critic observation shape changed from `980` to `1020`.
- `RewardManager` contained `track_ee_orientation`.
- `MetricsManager` contained 20 active terms, including:
  - `ee_target_quat_w/x/y/z`
  - `ee_orientation_error`
- Smoke log was removed after validation.

Important compatibility note:

- This changes the actor / critic input dimensions.
- Old checkpoints from `baseline_arm_ee` are not compatible with the new
  `baseline_arm_ee_pose` configuration.
- Start a new training run for the quaternion command version.

## 2026-05-19: GitHub Repository Preparation

### Problem

Needed to upload the project to GitHub without accidentally committing large
training outputs or local environment files.

### Idea

Track code, config, scripts, MJCF assets, and lockfile. Ignore local virtual
environment, logs, checkpoints, ONNX exports, TensorBoard events, and zip files.

### Changes

Files changed:

- `.gitignore`

Main ignored paths / patterns:

- `.venv/`
- `logs/`
- `*.pt`
- `*.onnx`
- `events.out.tfevents.*`
- `*.zip`
- Python cache and build outputs

### Test Result

`git status --ignored --short` showed ignored files as expected:

```text
!! .venv/
!! go2arm.zip
!! logs/
!! src/go2arm_mjlab/**/__pycache__/
```

GitHub remote:

```text
https://github.com/ChaimChen-CQ/Go2_Arx5_Mjlab_Chaim.git
```

Push initially failed with HTTP 408 timeout, likely due to network connection
instability. Local working tree was clean after commit.
