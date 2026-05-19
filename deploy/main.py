"""Entry point for Go2Arm sim-to-sim deployment experiments.

This mirrors the controller-oriented layout from rl-deploy-with-python while
keeping simulator-specific IO outside the controller core.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from controllers import Go2ArmController
from remote_ui import RemoteControlServer
from sim_adapters import MujocoSimAdapter


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run a Go2Arm deploy controller skeleton.")
  parser.add_argument(
    "--config",
    type=Path,
    default=Path(__file__).resolve().parent / "configs" / "go2arm_ee_position.yaml",
    help="Path to the deploy YAML config.",
  )
  parser.add_argument(
    "--policy",
    type=Path,
    default=None,
    help="Optional .pt, TorchScript, or ONNX policy path. Overrides config paths.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Load config and print expected deploy dimensions without opening a simulator.",
  )
  parser.add_argument(
    "--sim",
    choices=("none", "mujoco"),
    default="mujoco",
    help="Simulator adapter to run after loading the controller.",
  )
  parser.add_argument(
    "--steps",
    type=int,
    default=500,
    help="Number of deploy control steps to run for sim adapters. Use 0 to run until closed.",
  )
  parser.add_argument(
    "--render",
    action="store_true",
    help="Open a MuJoCo viewer for visual sim-to-sim playback.",
  )
  parser.add_argument(
    "--keyboard",
    action="store_true",
    help="Read command keys from the terminal while the MuJoCo sim runs.",
  )
  parser.add_argument(
    "--remote-ui",
    action="store_true",
    help="Start a browser remote-control panel for base and EE commands.",
  )
  parser.add_argument(
    "--command-server",
    action="store_true",
    help="Start the local command server for deploy/robot-joystick.",
  )
  parser.add_argument(
    "--remote-host",
    default="127.0.0.1",
    help="Host for --remote-ui.",
  )
  parser.add_argument(
    "--remote-port",
    type=int,
    default=8765,
    help="Port for --remote-ui.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  controller = Go2ArmController(args.config, policy_path=args.policy)
  if args.dry_run:
    controller.print_summary()
    return
  if args.sim == "none":
    controller.print_summary()
    return
  if args.sim == "mujoco":
    repo_root = Path(__file__).resolve().parents[1]
    adapter = MujocoSimAdapter(controller, repo_root=repo_root)
    remote = None
    if args.remote_ui or args.command_server:
      remote = RemoteControlServer(adapter, host=args.remote_host, port=args.remote_port)
      remote.start()
    try:
      adapter.run(args.steps, render=args.render, keyboard=args.keyboard)
    except KeyboardInterrupt:
      print("Stopped by user.")
    finally:
      if remote is not None:
        remote.stop()
    return


if __name__ == "__main__":
  main()
