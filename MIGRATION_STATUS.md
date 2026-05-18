# Go2Arm mjlab Migration Status

## Current Baseline

- Task registered as `Go2Arm-Flat`.
- Robot asset uses the combined Go2 + ARX L5 MJCF at `assets/go2arm/models/go2arm.xml`.
- Asset layout is split into `assets/go2arm/models/` for XML and `assets/go2arm/meshes/` for OBJ meshes.
- Action space controls Go2 legs plus ARX L5 joints, ordered as:
  `FR`, `FL`, `RR`, `RL`, then `joint1` through `joint7`.
- Actor observations use the IsaacLab project's main proprioceptive structure:
  base angular velocity, joint position, joint velocity, last action, base velocity command,
  projected gravity, all with history length 10.
- Core locomotion rewards and terminations are active.

## Pending Manipulation Work

- Add `ee_pose` command, arm observations, and end-effector rewards.
- Add manipulation-specific rewards or constraints when training reach/grasp behaviors.

## Useful Commands

```bash
uv run go2arm-list-envs --keyword Go2Arm
uv run go2arm-train Go2Arm-Flat --env.scene.num-envs=128 --agent.max-iterations=10
uv run go2arm-play Go2Arm-Flat --checkpoint-file=<path-to-model.pt>
```
