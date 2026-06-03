import os
from pathlib import Path
import sys
import toml


PYPROJECT_FILE_PATH = "pyproject.toml"


def project_root() -> Path:
    """Root directory of the project."""
    return Path(os.path.dirname(__file__)).parent


def pyproject_file_path() -> Path:
    """Path to the pyproject.toml."""
    return project_root() / PYPROJECT_FILE_PATH


def get_rasa_version_from_pyproject(pyproject_file=None) -> str:
    """Fetch rasa version from pyproject."""
    if pyproject_file is None:
        pyproject_file = pyproject_file_path()

    try:
        data = toml.load(pyproject_file)
        rasa_oss_version = data["tool"]["poetry"]["version"]
        return rasa_oss_version
    except (FileNotFoundError, TypeError):
        print(f"Unable to fetch from {pyproject_file}: file not found.")
        sys.exit(1)
    except toml.TomlDecodeError:
        print(f"Unable to parse {pyproject_file}: incorrect TOML file.")
        sys.exit(1)


if __name__ == "__main__":
    print(get_rasa_version_from_pyproject())
