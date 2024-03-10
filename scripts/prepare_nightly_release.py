"""Prepare a Rasa nightly release.

- increases the version number to dev version provided
"""
import argparse
import os
import sys
import toml
from pathlib import Path
from typing import Text
from pep440_version_utils import Version, is_valid_version

VERSION_FILE_PATH = "rasa/version.py"

PYPROJECT_FILE_PATH = "pyproject.toml"


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments for the release script."""

    parser = argparse.ArgumentParser(description="prepare the next nightly release")
    parser.add_argument(
        "--next_version", type=str, help="Rasa nightly version number",
    )

    return parser


def project_root() -> Path:
    """Root directory of the project."""
    return Path(os.path.dirname(__file__)).parent


def write_version_file(path: Text, version: Version) -> None:
    """Dump a new version into the python version file."""

    version_file = project_root() / path

    with version_file.open("w") as f:
        f.write(
            f"# this file will automatically be changed,\n"
            f"# do not add anything but the version number here!\n"
            f'__version__ = "{version}"\n'
        )


def write_version_to_pyproject(pyproject_file_path: Text, version: Version) -> None:
    """Dump a new version into the pyproject.toml."""

    pyproject_file = project_root() / pyproject_file_path

    try:
        data = toml.load(pyproject_file)
        data["tool"]["poetry"]["name"] = "rasa-nightly"
        data["tool"]["poetry"]["version"] = str(version)
        data["tool"]["poetry"]["packages"] = [{"include" : "rasa"},]
        with pyproject_file.open("w", encoding="utf8") as f:
            toml.dump(data, f)
    except (FileNotFoundError, TypeError):
        print(f"Unable to update {pyproject_file}: file not found.")
        sys.exit(1)
    except toml.TomlDecodeError:
        print(f"Unable to parse {pyproject_file}: incorrect TOML file.")
        sys.exit(1)


def parse_next_version(version: Text) -> Version:
    """Find the next version as a proper semantic version string."""
    return Version(version)


def next_version(args: argparse.Namespace) -> Version:
    """Take cmdline args or ask the user for the next version and return semver."""
    return parse_next_version(args.next_version)


def print_done_message(version: Version) -> None:
    """Print final information for the user on what to do next."""

    print()
    print(f"\033[94m All done - changes for rasa nightly version {version} are ready! \033[0m")
    print()


def main(args: argparse.Namespace) -> None:
    """Start a nightly release preparation."""

    print(
        "The release script will temporarily change the version number to a nightly version. Let's go!"
    )

    version = next_version(args)

    write_version_file(VERSION_FILE_PATH, version)
    write_version_to_pyproject(PYPROJECT_FILE_PATH, version)

    print_done_message(version)


if __name__ == "__main__":
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    main(cmdline_args)
