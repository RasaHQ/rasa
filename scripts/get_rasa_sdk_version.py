"""Get `rasa-sdk` version from pyproject.toml."""

import os
from pathlib import Path
from typing import Text

import tomlkit as toml


def project_root() -> Path:
    """Root directory of the project."""
    return Path(os.path.dirname(__file__)).parent


def get_rasa_sdk_version() -> Text:
    """Find out what the referenced version of the Rasa SDK is."""
    dependencies_filename = "pyproject.toml"
    with open(project_root() / dependencies_filename) as f:
        toml_data = toml.load(f)

    try:
        sdk_version = toml_data["tool"]["poetry"]["dependencies"]["rasa-sdk"][0]
        if not isinstance(sdk_version, str):
            sdk_version = sdk_version["version"]
        return sdk_version.strip("^ ~")
    except (AttributeError, KeyError, IndexError):
        raise Exception(f"Failed to find Rasa SDK version in {dependencies_filename}")


if __name__ == "__main__":
    print(get_rasa_sdk_version())
