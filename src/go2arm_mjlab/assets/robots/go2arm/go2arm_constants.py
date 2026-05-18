"""Go2Arm robot constants for mjlab.

This robot config uses the combined Unitree Go2 + ARX L5 MJCF.
"""

from pathlib import Path

import mujoco

from go2arm_mjlab import ASSET_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2ARM_XML: Path = ASSET_PATH / "go2arm" / "models" / "go2arm.xml"
assert GO2ARM_XML.exists(), f"Missing MJCF: {GO2ARM_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  asset_dir = GO2ARM_XML.parent / meshdir if meshdir and meshdir != "." else GO2ARM_XML.parent
  for path in asset_dir.rglob("*"):
    if path.is_file():
      rel_path = path.relative_to(asset_dir).as_posix()
      key = f"{meshdir}/{rel_path}" if meshdir and meshdir != "." else rel_path
      assets[key] = path.read_bytes()
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2ARM_XML))
  spec.assets = get_assets(spec.meshdir)
  # mjlab adds vectorized actuator wrappers from GO2ARM_ARTICULATION below.
  # Drop XML actuators to avoid double-driving the same joints.
  for xml_actuator in tuple(spec.actuators):
    spec.delete(xml_actuator)
  for xml_key in tuple(spec.keys):
    spec.delete(xml_key)
  return spec


GO2ARM_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=35.0,
  damping=0.8,
  effort_limit=35.0,
  armature=0.01,
)
GO2ARM_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=35.0,
  damping=0.8,
  effort_limit=35.0,
  armature=0.01,
)
GO2ARM_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=35.0,
  damping=0.8,
  effort_limit=35.0,
  armature=0.02,
)
GO2ARM_ACTUATOR_ARM_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=("joint[1-3]",),
  stiffness=80.0,
  damping=5.0,
  effort_limit=100.0,
  armature=0.005,
)
GO2ARM_ACTUATOR_ARM_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=("joint[4-6]",),
  stiffness=40.0,
  damping=5.0,
  effort_limit=100.0,
  armature=0.005,
)
GO2ARM_ACTUATOR_GRIPPER = BuiltinPositionActuatorCfg(
  target_names_expr=("joint7",),
  stiffness=40.0,
  damping=5.0,
  effort_limit=10.0,
  armature=0.005,
)


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.40),
  joint_pos={
    "FL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RL_hip_joint": 0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "FR_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
    "joint1": 0.0,
    "joint2": 1.55,
    "joint3": 0.95,
    "joint4": 0.45,
    "joint5": 0.0,
    "joint6": 0.0,
    "joint7": 0.044,
    "joint8": -0.044,
  },
  joint_vel={".*": 0.0},
)


_foot_regex = "^[FR][LR]$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  condim={_foot_regex: 3},
  priority={_foot_regex: 1},
  friction={_foot_regex: (1.0,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)


GO2ARM_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2ARM_ACTUATOR_HIP,
    GO2ARM_ACTUATOR_THIGH,
    GO2ARM_ACTUATOR_CALF,
    GO2ARM_ACTUATOR_ARM_SHOULDER,
    GO2ARM_ACTUATOR_ARM_WRIST,
    GO2ARM_ACTUATOR_GRIPPER,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2arm_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2ARM_ARTICULATION,
  )
