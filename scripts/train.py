"""Train mjlab tasks after registering Go2Arm tasks."""

import go2arm_mjlab.tasks  # noqa: F401
from mjlab.scripts.train import main


if __name__ == "__main__":
  main()
