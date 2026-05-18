"""Go2Arm flat-terrain task configuration for mjlab.

The current MJCF asset contains the Go2 base with an ARX L5 arm mounted on top.
This baseline still trains velocity locomotion; the arm is included in the
action and observation spaces and can be regularized/extended with manipulation
rewards later.
"""

import math

from go2arm_mjlab.assets.robots import get_go2arm_robot_cfg
from go2arm_mjlab.tasks.go2arm import mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


LEG_ACTION_ORDER = (
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
)
FRONT_LEG_ACTION_ORDER = (
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
)
ARM_ACTION_ORDER = (
  "joint1",
  "joint2",
  "joint3",
  "joint4",
  "joint5",
  "joint6",
  "joint7",
)
ACTION_ORDER = LEG_ACTION_ORDER + ARM_ACTION_ORDER

FOOT_NAMES = ("FR", "FL", "RR", "RL")
FOOT_SITE_NAMES = ("FR", "FL", "RR", "RL")
FOOT_GEOM_NAMES = FOOT_NAMES
BASE_BODY_NAME = "base"


def go2arm_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the first-stage Go2Arm flat task.

  This is the migration baseline: Go2 locomotion is live, while arm-specific
  command/reward hooks are intentionally left out until the arm MJCF is merged.
  """

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "velocity_commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "base_velocity"},
    ),
    "ee_position_commands": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "arm_ee_position"},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES)},
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=10,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=10,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=ACTION_ORDER,
      scale={
        ".*hip_joint": 0.25,
        ".*thigh_joint": 0.25,
        ".*calf_joint": 0.25,
        "joint[1-6]": 0.15,
        "joint7": 0.015,
      },
      clip={
        "joint1": (-3.14, 3.14),
        "joint2": (0.0, 3.14),
        "joint3": (0.0, 3.14),
        "joint4": (-1.7, 1.7),
        "joint5": (-1.7, 1.7),
        "joint6": (-3.14, 3.14),
        "joint7": (0.0, 0.044),
      },
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.1,
      heading_command=False,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.1, 0.8),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.5, 0.5),
        heading=None,
      ),
    ),
    "arm_ee_position": mdp.UniformEndEffectorCommandCfg(
      entity_name="robot",
      resampling_time_range=(2.0, 4.0),
      ranges=mdp.UniformEndEffectorCommandCfg.Ranges(
        x=(0.20, 0.45),
        y=(-0.25, 0.25),
        z=(0.15, 0.45),
      ),
    ),
  }

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.0), "yaw": (-3.14, 3.14)},
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(10.0, 15.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
  }

  if play:
    events.pop("push_robot", None)

  rewards = {
    "tracking_lin_vel_x_l1": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.5,
      params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.5,
      params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    ),
    "flat_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-3.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,))},
    ),
    "base_height_l2": RewardTermCfg(
      func=mdp.base_height_l2,
      weight=-5.0,
      params={"target_height": 0.40},
    ),
    "arm_posture_l2": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_ACTION_ORDER)},
    ),
    "front_leg_posture_l2": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-0.35,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=FRONT_LEG_ACTION_ORDER)},
    ),
    "track_ee_position": RewardTermCfg(
      func=mdp.track_end_effector_position,
      weight=1.0,
      params={
        "command_name": "arm_ee_position",
        "std": math.sqrt(0.05),
        "asset_cfg": SceneEntityCfg("robot", site_names=("end_effector",)),
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.02,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY_NAME,))},
    ),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.5,
        "command_name": "base_velocity",
        "command_threshold": 0.1,
      },
    ),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5, 0.5, 0.0],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "base_velocity",
        "sensor_name": "feet_ground_contact",
      },
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-0.0,
      params={
        "target_height": 0.08,
        "command_name": "base_velocity",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITE_NAMES),
      },
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(45.0)},
    ),
  }

  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "base_velocity",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (0.1, 0.35), "lin_vel_y": (-0.1, 0.1), "ang_vel_z": (-0.1, 0.1)},
          {"step": 1000 * 24, "lin_vel_x": (0.1, 0.8), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.5, 0.5)},
        ],
      },
    ),
  }

  if play:
    curriculum = {}
    events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_go2arm_robot_cfg()},
      sensors=(feet_ground_cfg,),
      num_envs=1,
      extent=2.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=BASE_BODY_NAME,
      distance=1.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
    ),
    decimation=4,
    episode_length_s=20.0 if not play else int(1e9),
  )
