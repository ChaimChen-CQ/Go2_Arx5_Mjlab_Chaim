from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def site_position_axis(
  env: ManagerBasedRlEnv,
  axis: int,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log one world-frame position axis for a selected site."""
  asset: Entity = env.scene[asset_cfg.name]
  pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :].squeeze(1)
  return pos_w[:, axis]


def site_target_position_axis(
  env: ManagerBasedRlEnv,
  axis: int,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log one world-frame target position axis for a base-frame site command."""
  asset: Entity = env.scene[asset_cfg.name]
  command_b = env.command_manager.get_command(command_name)
  assert command_b is not None, f"Command '{command_name}' not found."
  target_w = asset.data.root_link_pos_w + quat_apply(asset.data.root_link_quat_w, command_b[:, :3])
  return target_w[:, axis]


def site_position_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log end-effector position tracking error in meters."""
  asset: Entity = env.scene[asset_cfg.name]
  command_b = env.command_manager.get_command(command_name)
  assert command_b is not None, f"Command '{command_name}' not found."
  target_w = asset.data.root_link_pos_w + quat_apply(asset.data.root_link_quat_w, command_b[:, :3])
  pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :].squeeze(1)
  return torch.norm(target_w - pos_w, dim=1)


def site_quat_axis(
  env: ManagerBasedRlEnv,
  axis: int,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log one world-frame quaternion component for a selected site."""
  asset: Entity = env.scene[asset_cfg.name]
  quat_w = asset.data.site_quat_w[:, asset_cfg.site_ids, :].squeeze(1)
  return quat_w[:, axis]


def site_target_quat_axis(
  env: ManagerBasedRlEnv,
  axis: int,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log one world-frame target quaternion component for a base-frame command."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target_quat_w = quat_mul(asset.data.root_link_quat_w, command[:, 3:7])
  return target_quat_w[:, axis]


def site_orientation_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log end-effector orientation tracking error in radians."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target_quat_w = quat_mul(asset.data.root_link_quat_w, command[:, 3:7])
  quat_w = asset.data.site_quat_w[:, asset_cfg.site_ids, :].squeeze(1)
  return quat_error_magnitude(target_quat_w, quat_w)


def site_rpy_axis(
  env: ManagerBasedRlEnv,
  axis: int,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Log one world-frame roll/pitch/yaw axis for a selected site."""
  asset: Entity = env.scene[asset_cfg.name]
  quat_w = asset.data.site_quat_w[:, asset_cfg.site_ids, :].squeeze(1)
  rpy = euler_xyz_from_quat(quat_w)
  return rpy[axis]
