"""Headless MuJoCo adapter for Go2Arm sim-to-sim bringup."""

from __future__ import annotations

from contextlib import contextmanager
import itertools
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Any

import mujoco
import numpy as np

from controllers import Go2ArmController, RobotState


KEYBOARD_HELP = """
Keyboard controls
-----------------
Base velocity:
  W / S : lin_vel_x +/-
  A / D : lin_vel_y +/-
  Q / E : yaw rate +/-
  Space : zero base velocity

End-effector position:
  I / K : ee_x +/-
  J / L : ee_y +/-
  U / O : ee_z +/-

Other:
  C : reset EE command
  V : print current command
  H : print this help
  Esc or Ctrl-C : quit
"""


class MujocoSimAdapter:
  """Connect `Go2ArmController` to the standalone Go2Arm MJCF."""

  def __init__(self, controller: Go2ArmController, repo_root: Path):
    self.controller = controller
    self.cfg: dict[str, Any] = controller.cfg
    self.repo_root = Path(repo_root)

    model_path = Path(self.cfg["sim"]["mujoco_model"])
    if not model_path.is_absolute():
      model_path = self.repo_root / model_path
    self.model = self._load_model(model_path)
    self.data = mujoco.MjData(self.model)

    self.base_body_id = self._body_id(self.cfg["sim"]["base_body"])
    self.imu_ang_vel_sensor_id = self._sensor_id(self.cfg["sim"]["imu_ang_vel_sensor"])
    self.joint_qpos_addr = np.array(
      [self.model.jnt_qposadr[self._joint_id(name)] for name in controller.controlled_joints],
      dtype=np.int32,
    )
    self.joint_qvel_addr = np.array(
      [self.model.jnt_dofadr[self._joint_id(name)] for name in controller.controlled_joints],
      dtype=np.int32,
    )
    self.actuator_ids = np.array(
      [self._actuator_id_for_joint(name) for name in controller.controlled_joints],
      dtype=np.int32,
    )
    self.gripper_actuator_targets = self._build_gripper_targets()

    self.base_velocity_command = np.asarray(
      self.cfg["commands"]["base_velocity"]["default"], dtype=np.float32
    )
    self.default_base_velocity_command = self.base_velocity_command.copy()
    self.ee_position_command = np.asarray(
      self.cfg["commands"]["arm_ee_position"]["default"], dtype=np.float32
    )
    self.default_ee_position_command = self.ee_position_command.copy()
    self.command_lock = threading.Lock()
    self.stop_event = threading.Event()
    self.gait_period = float(self.cfg["gait"]["period"])
    self.loop_dt = float(self.cfg["control"]["loop_dt"])
    self.sim_steps_per_control = max(1, round(self.loop_dt / self.model.opt.timestep))

    self.reset()

  def reset(self) -> None:
    key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
      mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
    else:
      mujoco.mj_resetData(self.model, self.data)
      self.data.qpos[0:3] = np.array([0.0, 0.0, 0.40])
      self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
      for joint_name, value in self.cfg["default_joint_pos"].items():
        joint_id = self._joint_id(joint_name)
        self.data.qpos[self.model.jnt_qposadr[joint_id]] = float(value)
    for joint_name, value in self.cfg["sim"].get("hold_gripper", {}).items():
      joint_id = self._joint_id(joint_name)
      self.data.qpos[self.model.jnt_qposadr[joint_id]] = float(value)
    mujoco.mj_forward(self.model, self.data)

  def run(self, steps: int, render: bool = False, keyboard: bool = False) -> None:
    input_thread = self.start_keyboard_thread(keyboard)
    try:
      if render:
        self.run_rendered(steps, keyboard=keyboard)
        return
      for step in self._step_indices(steps):
        if self.stop_event.is_set():
          break
        targets = self.control_step()
        self.step_sim()
        if step % 50 == 0:
          self.print_step_summary(step, targets)
    finally:
      self.stop_keyboard_thread(input_thread)

  def run_rendered(self, steps: int, keyboard: bool = False) -> None:
    try:
      import mujoco.viewer
    except ImportError as exc:
      raise ImportError("mujoco.viewer is required for --render.") from exc

    key_callback = self.handle_viewer_key if keyboard else None
    with mujoco.viewer.launch_passive(
      self.model,
      self.data,
      key_callback=key_callback,
    ) as viewer:
      self.configure_viewer(viewer)
      if keyboard:
        print("Keyboard control also works when the MuJoCo window has focus.")
      for step in self._step_indices(steps):
        if self.stop_event.is_set():
          break
        step_start = time.time()
        targets = self.control_step()
        self.step_sim()
        if step % 50 == 0:
          self.print_step_summary(step, targets)
        viewer.sync()
        sleep_time = self.loop_dt - (time.time() - step_start)
        if sleep_time > 0.0:
          time.sleep(sleep_time)
        if not viewer.is_running():
          break

  def control_step(self) -> np.ndarray:
    state = self.read_state()
    with self.command_lock:
      base_velocity_command = self.base_velocity_command.copy()
      ee_position_command = self.ee_position_command.copy()
    action = self.controller.compute_action(
      state=state,
      base_velocity_command=base_velocity_command,
      ee_position_command=ee_position_command,
      gait_phase=self.gait_phase(),
    )
    targets = self.controller.action_to_joint_targets(action)
    self.write_joint_targets(targets)
    return targets

  def read_state(self) -> RobotState:
    return RobotState(
      base_ang_vel=self._sensor_data(self.imu_ang_vel_sensor_id),
      projected_gravity=self.projected_gravity(),
      joint_pos=self.data.qpos[self.joint_qpos_addr].copy(),
      joint_vel=self.data.qvel[self.joint_qvel_addr].copy(),
    )

  def write_joint_targets(self, targets: np.ndarray) -> None:
    self.data.ctrl[self.actuator_ids] = np.asarray(targets, dtype=np.float64)
    for actuator_id, target in self.gripper_actuator_targets.items():
      self.data.ctrl[actuator_id] = target

  def step_sim(self) -> None:
    for _ in range(self.sim_steps_per_control):
      mujoco.mj_step(self.model, self.data)

  def _step_indices(self, steps: int):
    if steps <= 0:
      return itertools.count()
    return range(steps)

  def gait_phase(self) -> np.ndarray:
    with self.command_lock:
      base_velocity_command = self.base_velocity_command.copy()
    if np.linalg.norm(base_velocity_command) < 0.1:
      return np.zeros(2, dtype=np.float32)
    phase = (self.data.time / self.gait_period) % 1.0
    return np.array(
      [np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)],
      dtype=np.float32,
    )

  def projected_gravity(self) -> np.ndarray:
    quat_wxyz = self.data.xquat[self.base_body_id]
    return self._quat_apply_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0]))

  def print_step_summary(self, step: int, targets: np.ndarray) -> None:
    base_height = self.data.qpos[2]
    target_preview = ", ".join(f"{value:.3f}" for value in targets[:3])
    print(
      f"step={step:05d} time={self.data.time:.3f} "
      f"base_z={base_height:.3f} first_targets=[{target_preview}]"
    )

  def configure_viewer(self, viewer: Any) -> None:
    viewer.cam.lookat[:] = self.data.qpos[0:3]
    viewer.cam.distance = 1.8
    viewer.cam.azimuth = 90.0
    viewer.cam.elevation = -15.0

  def start_keyboard_thread(self, enabled: bool) -> threading.Thread | None:
    if not enabled:
      return None
    if not sys.stdin.isatty():
      print("Keyboard control disabled: stdin is not a TTY.")
      return None
    print(KEYBOARD_HELP)
    print("Focus this terminal or the MuJoCo window for command keys.")
    self.print_command()

    def read_keys() -> None:
      with self.terminal_raw_mode():
        while not self.stop_event.is_set():
          ready, _, _ = select.select([sys.stdin], [], [], 0.05)
          if not ready:
            continue
          key = sys.stdin.read(1)
          if not key:
            continue
          if not self.handle_key(key):
            self.stop_event.set()
            break

    thread = threading.Thread(target=read_keys, daemon=True)
    thread.start()
    return thread

  def stop_keyboard_thread(self, thread: threading.Thread | None) -> None:
    self.stop_event.set()
    if thread is not None:
      thread.join(timeout=1.0)

  @contextmanager
  def terminal_raw_mode(self):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
      tty.setcbreak(fd)
      yield
    finally:
      termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

  def handle_key(self, key: str) -> bool:
    key = key.lower()
    changed = True
    should_continue = True
    with self.command_lock:
      if key == "w":
        self.base_velocity_command[0] = self._step_base_command(0, 1.0)
      elif key == "s":
        self.base_velocity_command[0] = self._step_base_command(0, -1.0)
      elif key == "a":
        self.base_velocity_command[1] = self._step_base_command(1, 1.0)
      elif key == "d":
        self.base_velocity_command[1] = self._step_base_command(1, -1.0)
      elif key == "q":
        self.base_velocity_command[2] = self._step_base_command(2, 1.0)
      elif key == "e":
        self.base_velocity_command[2] = self._step_base_command(2, -1.0)
      elif key == " ":
        self.base_velocity_command[:] = 0.0
      elif key == "i":
        self.ee_position_command[0] = self._step_ee_command(0, 1.0)
      elif key == "k":
        self.ee_position_command[0] = self._step_ee_command(0, -1.0)
      elif key == "j":
        self.ee_position_command[1] = self._step_ee_command(1, 1.0)
      elif key == "l":
        self.ee_position_command[1] = self._step_ee_command(1, -1.0)
      elif key == "u":
        self.ee_position_command[2] = self._step_ee_command(2, 1.0)
      elif key == "o":
        self.ee_position_command[2] = self._step_ee_command(2, -1.0)
      elif key == "c":
        self.ee_position_command[:] = self.default_ee_position_command
      elif key == "v":
        changed = False
        self.print_command()
      elif key == "h":
        changed = False
        print(KEYBOARD_HELP)
      elif key in ("\x03", "\x04", "\x1b"):
        changed = False
        should_continue = False
      else:
        changed = False

      if changed:
        self.print_command()
    return should_continue

  def handle_viewer_key(self, keycode: int) -> None:
    if keycode == 256:  # GLFW_KEY_ESCAPE
      self.stop_event.set()
      return
    if keycode == 32:
      key = " "
    elif 0 <= keycode <= 255:
      key = chr(keycode)
    else:
      return
    if not self.handle_key(key):
      self.stop_event.set()

  def print_command(self) -> None:
    print(
      "command | "
      f"vel=({self.base_velocity_command[0]:+.2f}, "
      f"{self.base_velocity_command[1]:+.2f}, "
      f"{self.base_velocity_command[2]:+.2f}) "
      f"ee=({self.ee_position_command[0]:.3f}, "
      f"{self.ee_position_command[1]:+.3f}, "
      f"{self.ee_position_command[2]:.3f})"
    )

  def command_state(self) -> dict[str, Any]:
    with self.command_lock:
      base = self.base_velocity_command.astype(float).tolist()
      ee = self.ee_position_command.astype(float).tolist()
    return {
      "base": base,
      "ee": ee,
      "time": float(self.data.time),
      "base_z": float(self.data.qpos[2]),
    }

  def apply_remote_command(self, payload: dict[str, Any]) -> None:
    with self.command_lock:
      if "base" in payload:
        base = payload["base"]
        fields = self.cfg["commands"]["base_velocity"]["fields"]
        for index, field in enumerate(fields):
          if field in base:
            self.base_velocity_command[index] = self._clamp_base_command(
              index, float(base[field])
            )
      if "ee" in payload:
        ee = payload["ee"]
        fields = self.cfg["commands"]["arm_ee_position"]["fields"]
        for index, field in enumerate(fields):
          if field in ee:
            self.ee_position_command[index] = self._clamp_ee_command(
              index, float(ee[field])
            )
      if payload.get("zero_base"):
        self.base_velocity_command[:] = 0.0
      if payload.get("reset_ee"):
        self.ee_position_command[:] = self.default_ee_position_command

  def _step_base_command(self, index: int, direction: float) -> float:
    command_cfg = self.cfg["commands"]["base_velocity"]
    field = command_cfg["fields"][index]
    step = float(command_cfg["steps"][field])
    lo, hi = command_cfg["limits"][field]
    value = float(self.base_velocity_command[index] + direction * step)
    return min(max(value, float(lo)), float(hi))

  def _step_ee_command(self, index: int, direction: float) -> float:
    command_cfg = self.cfg["commands"]["arm_ee_position"]
    field = command_cfg["fields"][index]
    step = float(command_cfg["steps"][field])
    lo, hi = command_cfg["limits"][field]
    value = float(self.ee_position_command[index] + direction * step)
    return min(max(value, float(lo)), float(hi))

  def _clamp_base_command(self, index: int, value: float) -> float:
    command_cfg = self.cfg["commands"]["base_velocity"]
    field = command_cfg["fields"][index]
    lo, hi = command_cfg["limits"][field]
    return min(max(value, float(lo)), float(hi))

  def _clamp_ee_command(self, index: int, value: float) -> float:
    command_cfg = self.cfg["commands"]["arm_ee_position"]
    field = command_cfg["fields"][index]
    lo, hi = command_cfg["limits"][field]
    return min(max(value, float(lo)), float(hi))

  def _sensor_data(self, sensor_id: int) -> np.ndarray:
    adr = self.model.sensor_adr[sensor_id]
    dim = self.model.sensor_dim[sensor_id]
    return self.data.sensordata[adr : adr + dim].copy()

  def _build_gripper_targets(self) -> dict[int, float]:
    targets: dict[int, float] = {}
    for joint_name, value in self.cfg["sim"].get("hold_gripper", {}).items():
      actuator_id = self._actuator_id_for_joint(joint_name)
      targets[actuator_id] = float(value)
    return targets

  def _actuator_id_for_joint(self, joint_name: str) -> int:
    joint_id = self._joint_id(joint_name)
    for actuator_id in range(self.model.nu):
      if int(self.model.actuator_trnid[actuator_id, 0]) == joint_id:
        return actuator_id
    raise KeyError(f"No actuator found for joint {joint_name}")

  def _joint_id(self, name: str) -> int:
    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise KeyError(f"Unknown MuJoCo joint: {name}")
    return joint_id

  def _body_id(self, name: str) -> int:
    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      raise KeyError(f"Unknown MuJoCo body: {name}")
    return body_id

  def _sensor_id(self, name: str) -> int:
    sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
      raise KeyError(f"Unknown MuJoCo sensor: {name}")
    return sensor_id

  def _load_model(self, model_path: Path) -> mujoco.MjModel:
    sim_cfg = self.cfg["sim"]
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.option.timestep = float(sim_cfg.get("physics_dt", 0.005))
    spec.option.iterations = int(sim_cfg.get("iterations", 10))
    spec.option.ls_iterations = int(sim_cfg.get("ls_iterations", 20))
    if bool(sim_cfg.get("disable_nativeccd", True)):
      spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)

    if bool(sim_cfg.get("add_floor", True)):
      spec.visual.headlight.active = 1
      spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
      spec.visual.headlight.diffuse = [0.65, 0.65, 0.65]
      spec.visual.headlight.specular = [0.0, 0.0, 0.0]
      spec.visual.rgba.haze = [0.92, 0.96, 1.0, 1.0]

      grid_texture = spec.add_texture()
      grid_texture.name = "deploy_grid"
      grid_texture.type = mujoco.mjtTexture.mjTEXTURE_2D
      grid_texture.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
      grid_texture.width = 256
      grid_texture.height = 256
      grid_texture.rgb1 = [0.18, 0.26, 0.34]
      grid_texture.rgb2 = [0.33, 0.45, 0.56]

      grid_material = spec.add_material()
      grid_material.name = "deploy_grid"
      grid_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "deploy_grid"
      grid_material.texrepeat = [8.0, 8.0]
      grid_material.texuniform = True
      grid_material.reflectance = 0.0

      floor = spec.worldbody.add_geom()
      floor.name = "floor"
      floor.type = mujoco.mjtGeom.mjGEOM_PLANE
      floor.size = [20.0, 20.0, 0.05]
      floor.pos = [0.0, 0.0, 0.0]
      floor.material = "deploy_grid"
      floor.friction = [1.0, 0.02, 0.01]
      floor.condim = 6
      floor.priority = 1

      key_light = spec.worldbody.add_light()
      key_light.name = "deploy_key_light"
      key_light.pos = [0.0, 0.0, 6.0]
      key_light.dir = [0.0, 0.0, -1.0]
      key_light.diffuse = [0.9, 0.9, 0.88]
      key_light.specular = [0.0, 0.0, 0.0]

      fill_light = spec.worldbody.add_light()
      fill_light.name = "deploy_fill_light"
      fill_light.pos = [0.0, -2.0, 4.5]
      fill_light.dir = [0.0, 0.3, -1.0]
      fill_light.diffuse = [0.25, 0.28, 0.3]
      fill_light.specular = [0.0, 0.0, 0.0]

    return spec.compile()

  @staticmethod
  def _quat_apply_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz
    q_vec_inv = -np.array([x, y, z], dtype=np.float64)
    uv = np.cross(q_vec_inv, vec)
    uuv = np.cross(q_vec_inv, uv)
    rotated = vec + 2.0 * (w * uv + uuv)
    return rotated.astype(np.float32)
