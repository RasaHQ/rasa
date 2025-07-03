import os
import sys
from pathlib import Path

VERSION_FILE_PATH = "rasa/version.py"

def get_current_version() -> str:
    """Return the current library version."""
    version_file_path = Path(os.path.dirname(__file__)).parent / VERSION_FILE_PATH
    if not version_file_path.is_file():
        raise FileNotFoundError(
            f"Failed to find version file at {version_file_path.absolute()}"
        )

    # context in which we evaluate the version py -
    # to be able to access the defined version, it already needs to live in the
    # context passed to exec
    _globals = {"__version__": ""}
    with version_file_path.open() as f:
        exec(f.read(), _globals)

    return _globals["__version__"]

if __name__ == "__main__":
    try:
        version = get_current_version()
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"{version}")
    sys.exit(0)
