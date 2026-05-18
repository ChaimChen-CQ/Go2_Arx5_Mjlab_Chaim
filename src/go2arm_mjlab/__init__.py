from pathlib import Path

PACKAGE_PATH = Path(__file__).resolve().parent
PROJECT_PATH = PACKAGE_PATH.parents[1]
ASSET_PATH = PROJECT_PATH / "assets"


def main() -> None:
    print("Go2Arm mjlab migration package")
