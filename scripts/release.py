"""Prepare a Rasa OSS release.

- creates a release branch
- creates a new changelog section in CHANGELOG.rst based on all collected changes
- increases the version number
- pushes the new branch to GitHub
"""
import argparse
import os
import sys
from pathlib import Path
from subprocess import CalledProcessError, check_call, check_output
from typing import Text, Set

import questionary
import semantic_version
from semantic_version import Version

VERSION_FILE_PATH = "rasa/version.py"

REPO_BASE_URL = "https://github.com/RasaHQ/rasa"

RELEASE_BRANCH_PREFIX = "prepare-release-"


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments for the release script."""

    parser = argparse.ArgumentParser(description="prepare the next library release")
    parser.add_argument(
        "--next_version",
        type=str,
        help="Either next version number or 'major', 'minor', 'patch'",
    )

    return parser


def project_root() -> Path:
    """Root directory of the project."""
    return Path(os.path.dirname(__file__)).parent


def version_file_path() -> Path:
    """Path to the python file containing the version number."""
    return project_root() / VERSION_FILE_PATH


def write_version_file(version: Text) -> None:
    """Dump a new version into the python version file."""

    with version_file_path().open("w") as f:
        f.write(
            f"# this file will automatically be changed,\n"
            f"# do not add anything but the version number here!\n"
            f'__version__ = "{version}"\n'
        )
    check_call(["git", "add", str(version_file_path().absolute())])


def get_current_version() -> Text:
    """Return the current library version."""

    if not version_file_path().is_file():
        raise FileNotFoundError(
            f"Failed to find version file at {version_file_path().absolute()}"
        )

    # context in which we evaluate the version py -
    # to be able to access the defined version, it already needs to live in the
    # context passed to exec
    _globals = {"__version__": ""}
    with version_file_path().open() as f:
        exec(f.read(), _globals)

    return _globals["__version__"]


def confirm_version(version: Text) -> bool:
    """Allow the user to confirm the version number."""

    if version in git_existing_tags():
        confirmed = questionary.confirm(
            f"Tag with version '{version}' already exists, overwrite?", default=False
        ).ask()
    else:
        confirmed = questionary.confirm(
            f"Current version is '{get_current_version()}. "
            f"Is the next version '{version}' correct ?",
            default=True,
        ).ask()
    if confirmed:
        return True
    else:
        print("Aborting.")
        sys.exit(1)


def ask_version() -> Text:
    """Allow the user to confirm the version number."""

    def is_valid_version_number(v: Text) -> bool:
        # noinspection PyBroadException
        try:
            return v in {"major", "minor", "patch"} or Version.coerce(v) is not None
        except Exception:
            # "coerce" did fail, this is probably not a valid version number
            return False

    version = questionary.text(
        "What is the version number you want to release "
        "('major', 'minor', 'patch' or valid version number)?",
        validate=is_valid_version_number,
    ).ask()

    if version:
        return version
    else:
        print("Aborting.")
        sys.exit(1)


def get_rasa_sdk_version() -> Text:
    """Find out what the referenced version of the Rasa SDK is."""

    env_file = project_root() / "requirements.txt"

    with env_file.open() as f:
        for line in f:
            if "rasa-sdk" in line:
                version = line.split("=")[-1]
                return version.strip()
        else:
            raise Exception("Failed to find Rasa SDK version in requirements.txt")


def validate_code_is_release_ready(version: Text) -> None:
    """Make sure the code base is valid (e.g. Rasa SDK is up to date)."""

    sdk = get_rasa_sdk_version()
    sdk_version = (Version.coerce(sdk).major, Version.coerce(sdk).minor)
    rasa_version = (Version.coerce(version).major, Version.coerce(version).minor)

    if sdk_version != rasa_version:
        print()
        print(
            f"\033[91m There is a mismatch between the Rasa SDK version ({sdk}) "
            f"and the version you want to release ({version}). Before you can "
            f"release Rasa OSS, you need to release the SDK and update "
            f"the dependency. \033[0m"
        )
        print()
        sys.exit(1)


def git_existing_tags() -> Set[Text]:
    """Return all existing tags in the local git repo."""

    stdout = check_output(["git", "tag"])
    return set(stdout.decode().split("\n"))


def git_current_branch() -> Text:
    """Returns the current git branch of the local repo."""

    try:
        output = check_output(["git", "symbolic-ref", "--short", "HEAD"])
        return output.decode().strip()
    except CalledProcessError:
        # e.g. we are in detached head state
        return "master"


def create_release_branch(version: Text) -> Text:
    """Create a new branch for this release. Returns the branch name."""

    branch = f"{RELEASE_BRANCH_PREFIX}{version}"
    check_call(["git", "checkout", "-b", branch])
    return branch


def create_commit(version: Text) -> None:
    """Creates a git commit with all stashed changes."""
    check_call(["git", "commit", "-m", f"prepared release of version {version}"])


def push_changes() -> None:
    """Pushes the current branch to origin."""
    check_call(["git", "push", "origin", "HEAD"])


def ensure_clean_git() -> None:
    """Makes sure the current working git copy is clean."""

    try:
        check_call(["git", "diff-index", "--quiet", "HEAD", "--"])
    except CalledProcessError:
        print("Your git is not clean. Release script can only be run from a clean git.")
        sys.exit(1)


def parse_next_version(version: Text) -> Text:
    """Find the next version as a proper semantic version string."""
    if version == "major":
        return str(Version.coerce(get_current_version()).next_major())
    elif version == "minor":
        return str(Version.coerce(get_current_version()).next_minor())
    elif version == "patch":
        return str(Version.coerce(get_current_version()).next_patch())
    elif semantic_version.validate(version):
        return version
    else:
        raise Exception(f"Invalid version number '{cmdline_args.next_version}'.")


def next_version(args: argparse.Namespace) -> Text:
    """Take cmdline args or ask the user for the next version and return semver."""
    return parse_next_version(args.next_version or ask_version())


def generate_changelog(version: Text) -> None:
    """Call tonwcrier and create a changelog from all available changelog entries."""
    check_call(["towncrier", "--yes", "--version", version], cwd=str(project_root()))


def print_done_message(branch: Text, base: Text, version: Text) -> None:
    """Print final information for the user on what to do next."""

    pull_request_url = f"{REPO_BASE_URL}/compare/{base}...{branch}?expand=1"

    print()
    print(f"\033[94m All done - changes for version {version} are ready! \033[0m")
    print()
    print(f"Please open a PR on GitHub: {pull_request_url}")


def main(args: argparse.Namespace) -> None:
    """Start a release preparation."""

    print(
        "The release script will increase the version number, "
        "create a changelog and create a release branch. Let's go!"
    )

    ensure_clean_git()
    version = next_version(args)
    confirm_version(version)

    validate_code_is_release_ready(version)

    write_version_file(version)

    generate_changelog(version)
    base = git_current_branch()
    branch = create_release_branch(version)

    create_commit(version)
    push_changes()

    print_done_message(branch, base, version)


if __name__ == "__main__":
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    main(cmdline_args)
