"""Virtual joystick client for the Go2Arm deploy command server."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import select
import sys
import termios
import time
import tty
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8765"

BASE_FIELDS = ("lin_vel_x", "lin_vel_y", "ang_vel_z")
EE_FIELDS = ("x", "y", "z")

LIMITS = {
  "base": {
    "lin_vel_x": (-0.3, 0.8),
    "lin_vel_y": (-0.5, 0.5),
    "ang_vel_z": (-0.8, 0.8),
  },
  "ee": {
    "x": (0.20, 0.55),
    "y": (-0.30, 0.30),
    "z": (0.15, 0.55),
  },
}

STEPS = {
  "base": {
    "lin_vel_x": 0.05,
    "lin_vel_y": 0.05,
    "ang_vel_z": 0.05,
  },
  "ee": {
    "x": 0.02,
    "y": 0.02,
    "z": 0.02,
  },
}

HELP = """Go2Arm virtual joystick

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
  R : refresh from sim
  H : help
  Esc or Ctrl-C : quit
"""


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Virtual joystick for Go2Arm sim-to-sim.")
  parser.add_argument(
    "--url",
    default=DEFAULT_URL,
    help="Command server URL from deploy/main.py --command-server.",
  )
  parser.add_argument(
    "--terminal",
    action="store_true",
    help="Use the terminal key-control UI instead of the graphical joystick.",
  )
  parser.add_argument(
    "--status",
    action="store_true",
    help="Print one command-server state sample and exit.",
  )
  return parser.parse_args()


def fetch_state(url: str) -> dict[str, Any]:
  with urlopen(f"{url.rstrip('/')}/state", timeout=1.0) as response:
    return json.loads(response.read().decode("utf-8"))


def post_command(url: str, payload: dict[str, Any]) -> dict[str, Any]:
  request = Request(
    f"{url.rstrip('/')}/command",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
  )
  with urlopen(request, timeout=1.0) as response:
    return json.loads(response.read().decode("utf-8"))


def clamp(group: str, field: str, value: float) -> float:
  lo, hi = LIMITS[group][field]
  return min(max(value, lo), hi)


def state_to_command(state: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
  base = {field: float(state["base"][index]) for index, field in enumerate(BASE_FIELDS)}
  ee = {field: float(state["ee"][index]) for index, field in enumerate(EE_FIELDS)}
  return base, ee


def draw(state: dict[str, Any], message: str = "") -> None:
  base, ee = state_to_command(state)
  lines = [
    "\033[2J\033[H",
    "Go2Arm Virtual Joystick",
    "",
    f"sim time: {state.get('time', 0.0):7.2f}s   base z: {state.get('base_z', 0.0):.3f}m",
    "",
    "Base velocity",
    f"  W/S lin_vel_x : {base['lin_vel_x']:+.2f}",
    f"  A/D lin_vel_y : {base['lin_vel_y']:+.2f}",
    f"  Q/E yaw       : {base['ang_vel_z']:+.2f}",
    "",
    "End-effector position",
    f"  I/K x : {ee['x']:.3f}",
    f"  J/L y : {ee['y']:+.3f}",
    f"  U/O z : {ee['z']:.3f}",
    "",
    "Space zero base | C reset EE | R refresh | H help | Esc/Ctrl-C quit",
  ]
  if message:
    lines.extend(["", message])
  sys.stdout.write("\n".join(lines))
  sys.stdout.flush()


@contextmanager
def terminal_cbreak():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setcbreak(fd)
    yield
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    sys.stdout.write("\033[?25h\n")
    sys.stdout.flush()


def apply_key(url: str, state: dict[str, Any], key: str) -> tuple[dict[str, Any], bool, str]:
  base, ee = state_to_command(state)
  payload: dict[str, Any] = {}
  key = key.lower()

  if key == "\x1b" or key == "\x03":
    return state, False, "quit"
  if key == "h":
    return state, True, HELP
  if key == "r":
    return fetch_state(url), True, "refreshed"
  if key == " ":
    payload["base"] = {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_z": 0.0}
  elif key == "c":
    payload["reset_ee"] = True
  elif key in ("w", "s"):
    sign = 1.0 if key == "w" else -1.0
    field = "lin_vel_x"
    payload["base"] = {field: clamp("base", field, base[field] + sign * STEPS["base"][field])}
  elif key in ("a", "d"):
    sign = 1.0 if key == "a" else -1.0
    field = "lin_vel_y"
    payload["base"] = {field: clamp("base", field, base[field] + sign * STEPS["base"][field])}
  elif key in ("q", "e"):
    sign = 1.0 if key == "q" else -1.0
    field = "ang_vel_z"
    payload["base"] = {field: clamp("base", field, base[field] + sign * STEPS["base"][field])}
  elif key in ("i", "k"):
    sign = 1.0 if key == "i" else -1.0
    field = "x"
    payload["ee"] = {field: clamp("ee", field, ee[field] + sign * STEPS["ee"][field])}
  elif key in ("j", "l"):
    sign = 1.0 if key == "j" else -1.0
    field = "y"
    payload["ee"] = {field: clamp("ee", field, ee[field] + sign * STEPS["ee"][field])}
  elif key in ("u", "o"):
    sign = 1.0 if key == "u" else -1.0
    field = "z"
    payload["ee"] = {field: clamp("ee", field, ee[field] + sign * STEPS["ee"][field])}
  else:
    return state, True, ""

  return post_command(url, payload), True, "sent"


class GraphicalJoystick:
  def __init__(self, url: str):
    try:
      import tkinter as tk
      from tkinter import ttk
    except Exception as exc:
      raise SystemExit(
        f"Cannot start graphical joystick: {exc}\n"
        "Use ./deploy/robot-joystick --terminal as a fallback."
      ) from exc

    self.tk = tk
    self.ttk = ttk
    self.url = url.rstrip("/")
    self.root = tk.Tk()
    self.root.title("Go2Arm Virtual Joystick")
    self.root.geometry("760x520")
    self.root.minsize(680, 460)

    self.status_var = tk.StringVar(value="connecting")
    self.message_var = tk.StringVar(value="")
    self.slider_vars: dict[tuple[str, str], Any] = {}
    self.value_vars: dict[tuple[str, str], Any] = {}
    self.updating = False

    self._build()
    self.root.bind("<KeyPress>", self._on_key)
    self._refresh()

  def run(self) -> None:
    self.root.mainloop()

  def _build(self) -> None:
    ttk = self.ttk
    root = self.root
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    header = ttk.Frame(root, padding=(18, 16, 18, 8))
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)
    ttk.Label(header, text="Go2Arm Virtual Joystick", font=("Sans", 18, "bold")).grid(
      row=0, column=0, sticky="w"
    )
    ttk.Label(header, textvariable=self.status_var).grid(row=0, column=1, sticky="e")

    body = ttk.Frame(root, padding=(18, 8, 18, 12))
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=1)
    body.rowconfigure(0, weight=1)

    base = ttk.LabelFrame(body, text="Base Velocity", padding=14)
    base.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    ee = ttk.LabelFrame(body, text="End Effector Position", padding=14)
    ee.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    self._add_slider(base, "base", "lin_vel_x", "W / S  x", -0.3, 0.8, 0)
    self._add_slider(base, "base", "lin_vel_y", "A / D  y", -0.5, 0.5, 1)
    self._add_slider(base, "base", "ang_vel_z", "Q / E  yaw", -0.8, 0.8, 2)
    ttk.Button(base, text="Zero Base", command=self._zero_base).grid(
      row=3, column=0, columnspan=3, sticky="ew", pady=(16, 0)
    )

    self._add_slider(ee, "ee", "x", "I / K  x", 0.20, 0.55, 0)
    self._add_slider(ee, "ee", "y", "J / L  y", -0.30, 0.30, 1)
    self._add_slider(ee, "ee", "z", "U / O  z", 0.15, 0.55, 2)
    ttk.Button(ee, text="Reset EE", command=self._reset_ee).grid(
      row=3, column=0, columnspan=3, sticky="ew", pady=(16, 0)
    )

    footer = ttk.Frame(root, padding=(18, 0, 18, 16))
    footer.grid(row=2, column=0, sticky="ew")
    footer.columnconfigure(0, weight=1)
    ttk.Label(
      footer,
      text="Keyboard: W/S A/D Q/E, I/K J/L U/O, Space zero base, C reset EE",
    ).grid(row=0, column=0, sticky="w")
    ttk.Label(footer, textvariable=self.message_var).grid(row=1, column=0, sticky="w")

  def _add_slider(
    self,
    parent: Any,
    group: str,
    field: str,
    label: str,
    lo: float,
    hi: float,
    row: int,
  ) -> None:
    ttk = self.ttk
    tk = self.tk
    parent.columnconfigure(1, weight=1)
    var = tk.DoubleVar(value=0.0)
    value_var = tk.StringVar(value="0.000")
    self.slider_vars[(group, field)] = var
    self.value_vars[(group, field)] = value_var

    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=8)
    scale = ttk.Scale(
      parent,
      from_=lo,
      to=hi,
      orient="horizontal",
      variable=var,
      command=lambda _value, g=group, f=field: self._slider_changed(g, f),
    )
    scale.grid(row=row, column=1, sticky="ew", padx=10, pady=8)
    ttk.Label(parent, textvariable=value_var, width=8).grid(row=row, column=2, sticky="e")

  def _slider_changed(self, group: str, field: str) -> None:
    if self.updating:
      return
    value = float(self.slider_vars[(group, field)].get())
    self.value_vars[(group, field)].set(f"{value:.3f}")
    self._post({group: {field: value}})

  def _post(self, payload: dict[str, Any]) -> None:
    try:
      state = post_command(self.url, payload)
    except Exception as exc:
      self.message_var.set(f"send failed: {exc}")
      return
    self._set_state(state, "sent")

  def _zero_base(self) -> None:
    self._post({"base": {"lin_vel_x": 0.0, "lin_vel_y": 0.0, "ang_vel_z": 0.0}})

  def _reset_ee(self) -> None:
    self._post({"reset_ee": True})

  def _refresh(self) -> None:
    try:
      state = fetch_state(self.url)
      self._set_state(state, "")
    except Exception as exc:
      self.status_var.set("disconnected")
      self.message_var.set(f"waiting for command server: {exc}")
    self.root.after(300, self._refresh)

  def _set_state(self, state: dict[str, Any], message: str) -> None:
    self.updating = True
    try:
      base, ee = state_to_command(state)
      for field, value in base.items():
        self.slider_vars[("base", field)].set(value)
        self.value_vars[("base", field)].set(f"{value:.3f}")
      for field, value in ee.items():
        self.slider_vars[("ee", field)].set(value)
        self.value_vars[("ee", field)].set(f"{value:.3f}")
      self.status_var.set(
        f"connected | sim t={state.get('time', 0.0):.2f}s | base z={state.get('base_z', 0.0):.3f}m"
      )
      self.message_var.set(message)
    finally:
      self.updating = False

  def _on_key(self, event: Any) -> None:
    key = event.char
    if event.keysym == "Escape":
      self.root.destroy()
      return
    if not key:
      return
    try:
      state = fetch_state(self.url)
      state, keep_running, message = apply_key(self.url, state, key)
    except Exception as exc:
      self.message_var.set(f"key command failed: {exc}")
      return
    if not keep_running:
      self.root.destroy()
      return
    self._set_state(state, message)


def run(url: str) -> None:
  try:
    state = fetch_state(url)
  except URLError as exc:
    raise SystemExit(
      f"Cannot connect to {url}. Start sim with --command-server first."
    ) from exc

  message = "connected"
  sys.stdout.write("\033[?25l")
  sys.stdout.flush()
  with terminal_cbreak():
    while True:
      draw(state, message)
      ready, _, _ = select.select([sys.stdin], [], [], 0.2)
      if not ready:
        try:
          state = fetch_state(url)
          message = ""
        except URLError:
          message = "waiting for command server..."
          time.sleep(0.2)
        continue
      key = sys.stdin.read(1)
      try:
        state, keep_running, message = apply_key(url, state, key)
      except URLError as exc:
        message = f"send failed: {exc}"
        keep_running = True
      if not keep_running:
        break


def main() -> None:
  args = parse_args()
  if args.status:
    print(json.dumps(fetch_state(args.url), indent=2))
    return
  if args.terminal:
    run(args.url)
    return
  GraphicalJoystick(args.url).run()


if __name__ == "__main__":
  main()
