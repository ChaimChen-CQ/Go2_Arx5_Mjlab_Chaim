# Go2Arm Deploy Commands

## Prepare Policy

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
cp logs/rsl_rl/go2arm_flat/2026-05-19_17-01-59_baseline_arm_ee_position_resume_1800/policy.onnx \
  deploy/policies/go2arm_ee_position.onnx
```

## Check Config

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
uv run python deploy/main.py --dry-run
```

## Run MuJoCo

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
uv run python deploy/main.py --sim mujoco --steps 3000 --render
```

## Run With Keyboard Control

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
uv run python deploy/main.py --sim mujoco --steps 10000 --render --keyboard
```

```text
W / S : base lin_vel_x +/-
A / D : base lin_vel_y +/-
Q / E : base yaw rate +/-
Space : zero base velocity
I / K : EE x +/-
J / L : EE y +/-
U / O : EE z +/-
C     : reset EE command
V     : print current command
H     : print help
Esc or Ctrl-C : quit
```

## Run With Browser UI

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
uv run python deploy/main.py --sim mujoco --steps 10000 --render --remote-ui
```

Open:

```text
http://127.0.0.1:8765
```

## Run With Virtual Joystick

Terminal 1:

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
uv run python deploy/main.py --sim mujoco --steps 0 --render --command-server
```

Terminal 2:

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
./deploy/robot-joystick
```

Terminal fallback:

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
./deploy/robot-joystick --terminal
```

## Run Specific Policy

```bash
cd /home/chaim/Go2_ArxL5_Mjlab_Chaim
uv run python deploy/main.py \
  --sim mujoco \
  --steps 3000 \
  --render \
  --policy deploy/policies/go2arm_ee_position.onnx
```
