"""List mjlab tasks after registering Go2Arm tasks."""

import go2arm_mjlab.tasks  # noqa: F401
from mjlab.scripts.list_envs import main

__all__ = ["main"]
