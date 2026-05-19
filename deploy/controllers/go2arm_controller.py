"""Go2Arm position-control deploy controller skeleton.

The structure follows rl-deploy-with-python:

- load a YAML deploy config
- load an optional ONNX policy
- keep an observation history buffer
- compute policy actions at the trained decimation rate
- convert scaled actions into default-offset joint position targets

Simulator-specific state collection and command publishing should live in a
thin adapter around this controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import yaml


@dataclass
class RobotState:
  """Minimal state needed to build the current Go2Arm actor observation."""

  base_ang_vel: np.ndarray
  projected_gravity: np.ndarray
  joint_pos: np.ndarray
  joint_vel: np.ndarray


class Go2ArmController:
  term_order = (
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
    "velocity_commands",
    "gait_phase",
    "ee_position_commands",
    "projected_gravity",
  )

  def __init__(self, config_path: Path, policy_path: Path | None = None):
    self.config_path = Path(config_path)
    self.deploy_root = self.config_path.parent.parent
    self.repo_root = self.deploy_root.parent
    with self.config_path.open("r") as f:
      self.cfg: dict[str, Any] = yaml.safe_load(f)

    self.policy_path = self._resolve_policy_path(policy_path)
    self.policy_session = None
    self.policy_input_name = None
    self.policy_output_names: list[str] = []
    self.policy_input_shape: tuple[Any, ...] | None = None
    self.policy_fn: Callable[[np.ndarray], np.ndarray] | None = None
    if self.policy_path is not None and self.policy_path.exists():
      self._load_policy(self.policy_path)

    policy_cfg = self.cfg["policy"]
    control_cfg = self.cfg["control"]
    self.action_dim = int(policy_cfg["action_dim"])
    self.actor_observation_dim = int(policy_cfg["actor_observation_dim"])
    self.history_length = int(policy_cfg["history_length"])
    self.decimation = int(control_cfg["decimation"])
    action_clip = policy_cfg.get("action_clip")
    self.action_clip = None if action_clip is None else float(action_clip)

    self.controlled_joints = list(self.cfg["controlled_joints"])
    self.default_joint_pos = np.array(
      [self.cfg["default_joint_pos"][name] for name in self.controlled_joints],
      dtype=np.float32,
    )
    self.action_scale = np.array(
      [self._scale_for_joint(name) for name in self.controlled_joints],
      dtype=np.float32,
    )

    if len(self.controlled_joints) != self.action_dim:
      raise ValueError(
        f"controlled_joints length {len(self.controlled_joints)} != action_dim {self.action_dim}"
      )

    self.last_action = np.zeros(self.action_dim, dtype=np.float32)
    self.obs_history = self._make_history_buffers()
    self.step_count = 0

  def print_summary(self) -> None:
    print(f"config: {self.config_path}")
    print(f"task_id: {self.cfg['task_id']}")
    print(f"run_name: {self.cfg['run_name']}")
    print(f"policy_path: {self.policy_path}")
    print(f"policy_loaded: {self.policy_fn is not None or self.policy_session is not None}")
    print(f"action_dim: {self.action_dim}")
    print(f"actor_observation_dim: {self.actor_observation_dim}")
    print(f"history_length: {self.history_length}")
    print("controlled_joints:")
    for index, name in enumerate(self.controlled_joints):
      print(f"  {index:02d}: {name}")

  def compute_action(
    self,
    state: RobotState,
    base_velocity_command: np.ndarray,
    ee_position_command: np.ndarray,
    gait_phase: np.ndarray,
  ) -> np.ndarray:
    """Compute scaled policy action.

    The caller must provide values in the same order and frame as training:

    - base angular velocity from `robot/imu_ang_vel`
    - relative joint positions for `controlled_joints`
    - relative joint velocities for `controlled_joints`
    - base velocity command `[lin_vel_x, lin_vel_y, ang_vel_z]`
    - gait phase `[sin, cos]`
    - EE position command `[x, y, z]`
    - projected gravity
    """
    obs = self._build_observation_terms(
      state=state,
      base_velocity_command=base_velocity_command,
      ee_position_command=ee_position_command,
      gait_phase=gait_phase,
    )
    self._push_observation_terms(obs)
    actor_obs = self.actor_observation()

    if self.policy_session is None and self.policy_fn is None:
      return np.zeros(self.action_dim, dtype=np.float32)

    if self.step_count % self.decimation == 0:
      policy_action = self._run_policy(actor_obs)
      if self.action_clip is not None:
        policy_action = np.clip(policy_action, -self.action_clip, self.action_clip)
      self.last_action = policy_action.astype(np.float32)
      if self.last_action.shape[0] != self.action_dim:
        raise ValueError(
          f"policy output dim {self.last_action.shape[0]} != action_dim {self.action_dim}"
        )

    self.step_count += 1
    return self.last_action.copy()

  def action_to_joint_targets(self, action: np.ndarray) -> np.ndarray:
    """Convert scaled policy action to default-offset joint position targets."""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != self.action_dim:
      raise ValueError(f"action dim {action.shape[0]} != action_dim {self.action_dim}")
    return self.default_joint_pos + action * self.action_scale

  def actor_observation(self) -> np.ndarray:
    """Pack history like mjlab: term history first, then concatenate terms."""
    obs = np.concatenate(
      [self.obs_history[name].reshape(-1) for name in self.term_order],
      axis=0,
    ).astype(np.float32)
    if obs.shape[0] != self.actor_observation_dim:
      raise ValueError(f"actor obs dim {obs.shape[0]} != expected {self.actor_observation_dim}")
    return obs

  def _build_observation_terms(
    self,
    state: RobotState,
    base_velocity_command: np.ndarray,
    ee_position_command: np.ndarray,
    gait_phase: np.ndarray,
  ) -> dict[str, np.ndarray]:
    joint_pos_rel = np.asarray(state.joint_pos, dtype=np.float32) - self.default_joint_pos
    joint_vel_rel = np.asarray(state.joint_vel, dtype=np.float32)
    return {
      "base_ang_vel": np.asarray(state.base_ang_vel, dtype=np.float32),
      "joint_pos": joint_pos_rel,
      "joint_vel": joint_vel_rel,
      "actions": self.last_action.astype(np.float32),
      "velocity_commands": np.asarray(base_velocity_command, dtype=np.float32),
      "gait_phase": np.asarray(gait_phase, dtype=np.float32),
      "ee_position_commands": np.asarray(ee_position_command, dtype=np.float32),
      "projected_gravity": np.asarray(state.projected_gravity, dtype=np.float32),
    }

  def _push_observation_terms(self, obs: dict[str, np.ndarray]) -> None:
    if self.step_count == 0:
      for name, value in obs.items():
        self.obs_history[name][:] = value
      return
    for name, value in obs.items():
      self.obs_history[name][:-1] = self.obs_history[name][1:]
      self.obs_history[name][-1] = value

  def _make_history_buffers(self) -> dict[str, np.ndarray]:
    dims = {
      "base_ang_vel": 3,
      "joint_pos": self.action_dim,
      "joint_vel": self.action_dim,
      "actions": self.action_dim,
      "velocity_commands": 3,
      "gait_phase": 2,
      "ee_position_commands": 3,
      "projected_gravity": 3,
    }
    total_dim = sum(dims[name] * self.history_length for name in self.term_order)
    if total_dim != self.actor_observation_dim:
      raise ValueError(f"configured term history dim {total_dim} != actor obs dim {self.actor_observation_dim}")
    return {
      name: np.zeros((self.history_length, dims[name]), dtype=np.float32)
      for name in self.term_order
    }

  def _run_policy(self, actor_obs: np.ndarray) -> np.ndarray:
    if self.policy_fn is not None:
      return self.policy_fn(actor_obs)
    if self.policy_session is None:
      return np.zeros(self.action_dim, dtype=np.float32)
    policy_input = actor_obs.astype(np.float32)
    if self.policy_input_shape is not None and len(self.policy_input_shape) == 2:
      policy_input = policy_input.reshape(1, -1)
    outputs = self.policy_session.run(
      self.policy_output_names,
      {self.policy_input_name: policy_input},
    )
    return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

  def _load_policy(self, policy_path: Path) -> None:
    if policy_path.suffix == ".onnx":
      self._load_onnx_policy(policy_path)
      return
    try:
      self.policy_fn = self._load_torch_policy(policy_path)
    except ImportError as exc:
      warnings.warn(f"{exc} Continuing with zero actions.", stacklevel=2)
      self.policy_fn = None

  def _load_onnx_policy(self, policy_path: Path) -> None:
    try:
      import onnxruntime as ort
    except ImportError as exc:
      raise ImportError(
        "onnxruntime is required for deploy policy inference. Install it in the "
        "deploy environment or run with --dry-run."
      ) from exc

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.policy_session = ort.InferenceSession(
      str(policy_path),
      sess_options=session_options,
      providers=["CPUExecutionProvider"],
    )
    self.policy_input_name = self.policy_session.get_inputs()[0].name
    self.policy_input_shape = tuple(self.policy_session.get_inputs()[0].shape)
    self.policy_output_names = [output.name for output in self.policy_session.get_outputs()]

  def _load_torch_policy(self, policy_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    try:
      import torch
      from torch import nn
    except ImportError as exc:
      raise ImportError("torch is required to load TorchScript or .pt checkpoints.") from exc

    try:
      scripted = torch.jit.load(policy_path, map_location="cpu")
      scripted.eval()

      def run_scripted(actor_obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
          output = scripted(torch.from_numpy(actor_obs).unsqueeze(0))
        return output.squeeze(0).cpu().numpy().astype(np.float32)

      return run_scripted
    except RuntimeError:
      checkpoint = torch.load(policy_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("actor_state_dict")
    if state_dict is None:
      raise ValueError(f"Could not find actor_state_dict in {policy_path}")

    weight_keys = sorted(
      [key for key in state_dict if key.startswith("mlp.") and key.endswith(".weight")],
      key=lambda key: int(key.split(".")[1]),
    )
    layers: list[nn.Module] = []
    remapped: dict[str, Any] = {}
    seq_idx = 0
    for i, weight_key in enumerate(weight_keys):
      layer_idx = weight_key.split(".")[1]
      weight = state_dict[f"mlp.{layer_idx}.weight"]
      bias = state_dict[f"mlp.{layer_idx}.bias"]
      layers.append(nn.Linear(weight.shape[1], weight.shape[0]))
      remapped[f"{seq_idx}.weight"] = weight
      remapped[f"{seq_idx}.bias"] = bias
      seq_idx += 1
      if i != len(weight_keys) - 1:
        layers.append(nn.ELU())
        seq_idx += 1

    mlp = nn.Sequential(*layers)
    mlp.load_state_dict(remapped)
    mlp.eval()
    mean = state_dict.get("obs_normalizer._mean")
    std = state_dict.get("obs_normalizer._std")
    if mean is None or std is None:
      mean = torch.zeros(self.actor_observation_dim)
      std = torch.ones(self.actor_observation_dim)
    mean = mean.float().reshape(-1)
    std = torch.clamp(std.float().reshape(-1), min=1.0e-6)

    def run_checkpoint(actor_obs: np.ndarray) -> np.ndarray:
      with torch.no_grad():
        obs = torch.from_numpy(actor_obs).float().reshape(-1)
        obs = (obs - mean) / std
        output = mlp(obs.unsqueeze(0))
      return output.squeeze(0).cpu().numpy().astype(np.float32)

    return run_checkpoint

  def _resolve_policy_path(self, policy_path: Path | None) -> Path | None:
    if policy_path is not None:
      return Path(policy_path)
    onnx_path = self._optional_path("policy", "onnx_path")
    if onnx_path is not None and onnx_path.exists():
      return onnx_path
    checkpoint = self.cfg.get("checkpoint")
    if checkpoint:
      path = Path(checkpoint)
      if not path.is_absolute():
        path = self.repo_root / path
      return path
    return onnx_path

  def _optional_path(self, section: str, key: str) -> Path | None:
    value = self.cfg.get(section, {}).get(key)
    if not value:
      return None
    path = Path(value)
    if not path.is_absolute():
      path = self.deploy_root / path
    return path

  def _scale_for_joint(self, joint_name: str) -> float:
    if joint_name.startswith(("F", "R")) and joint_name.endswith("_joint"):
      return float(self.cfg["action_scale"]["legs"])
    if joint_name.startswith("joint"):
      return float(self.cfg["action_scale"]["arm_joint1_to_joint6"])
    raise KeyError(f"No action scale rule for joint {joint_name}")
