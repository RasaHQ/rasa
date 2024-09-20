"""Prepare a Rasa Private release.

When run with `prepare`:
- creates a release branch
- creates a new changelog section in CHANGELOG.md based on all collected changes
- increases the version number
- pushes the new branch to GitHub

When run with `tag`:
- tags the current commit with the version number found in the version module
- pushes the tag to GitHub, which will kick off the release workflows.
When run with `fetch`:
- retrieves the current rasa-oss dependency version stored in pyproject.toml
"""

import argparse
import os
import re
import sys
from pathlib import Path
from subprocess import CalledProcessError, check_call, check_output
from typing import Text, Set

import questionary
import toml
from pep440_version_utils import Version, is_valid_version

VERSION_FILE_PATH = "rasa/version.py"

PYPROJECT_FILE_PATH = "pyproject.toml"

REPO_BASE_URL = "https://github.com/RasaHQ/rasa-private"

RELEASE_BRANCH_PREFIX = "prepare-release-"

PRE_RELEASE_BRANCH_PREFIX = "prepare-release-pre-"

PRERELEASE_FLAVORS = ("alpha", "beta", "rc")

RELEASE_BRANCH_PATTERN = re.compile(r"^\d+\.\d+\.x$")


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments for the release script."""
    parser = argparse.ArgumentParser(
        description="prepare or tag the next library release"
    )
    subparsers = parser.add_subparsers()
    prepare_subparser = subparsers.add_parser(
        "prepare",
        description="Prepare the next release",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare_subparser.set_defaults(func=prepare_release)
    prepare_subparser.add_argument(
        "--next_version",
        type=str,
        help=(
            "Either next version number or 'major', "
            "'minor', 'micro', 'alpha', 'beta', 'rc'"
        ),
    )
    prepare_subparser.add_argument(
        "--interactive",
        action="store_true",
    )
    tag_subparser = subparsers.add_parser(
        "tag",
        description="Tag the next release",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    tag_subparser.set_defaults(func=tag_release)
    tag_subparser.add_argument("--skip-confirmation", action="store_true")

    return parser


def project_root() -> Path:
    """Root directory of the project."""
    return Path(os.path.dirname(__file__)).parent


def version_file_path() -> Path:
    """Path to the python file containing the version number."""
    return project_root() / VERSION_FILE_PATH


def pyproject_file_path() -> Path:
    """Path to the pyproject.toml."""
    return project_root() / PYPROJECT_FILE_PATH


def write_version_file(version: Version) -> None:
    """Dump a new version into the python version file."""
    with version_file_path().open("w") as f:
        f.write(
            f"# this file will automatically be changed,\n"
            f"# do not add anything but the version number here!\n"
            f'__version__ = "{version}"\n'
        )
    check_call(["git", "add", str(version_file_path().absolute())])


def write_version_to_pyproject(version: Version) -> None:
    """Dump a new version into the pyproject.toml."""
    pyproject_file = pyproject_file_path()

    try:
        data = toml.load(pyproject_file)
        data["tool"]["poetry"]["version"] = str(version)
        with pyproject_file.open("w", encoding="utf8") as f:
            toml.dump(data, f)
    except (FileNotFoundError, TypeError):
        print(f"Unable to update {pyproject_file}: file not found.")
        sys.exit(1)
    except toml.TomlDecodeError:
        print(f"Unable to parse {pyproject_file}: incorrect TOML file.")
        sys.exit(1)

    check_call(["git", "add", str(pyproject_file.absolute())])


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


def confirm_version(version: Version) -> bool:
    """Allow the user to confirm the version number."""
    if str(version) in git_existing_tags():
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
        return v in {
            "major",
            "minor",
            "micro",
            "alpha",
            "beta",
            "rc",
        } or is_valid_version(v)

    current_version = Version(get_current_version())
    next_micro_version = str(current_version.next_micro())
    next_alpha_version = str(current_version.next_alpha())
    next_beta_version = str(current_version.next_beta())
    version = questionary.text(
        f"What is the version number you want to release "
        f"('major', 'minor', 'micro', 'alpha', 'beta', 'rc' or valid version number "
        f"e.g. '{next_micro_version}' or '{next_alpha_version}'"
        f" or '{next_beta_version}')?",
        validate=is_valid_version_number,
    ).ask()

    if version in PRERELEASE_FLAVORS and not current_version.pre:
        # at this stage it's hard to guess the kind of version bump the
        # releaser wants, so we ask them
        if version == "alpha":
            choices = [
                str(current_version.next_alpha("minor")),
                str(current_version.next_alpha("micro")),
                str(current_version.next_alpha("major")),
            ]
        elif version == "beta":
            choices = [
                str(current_version.next_beta("minor")),
                str(current_version.next_beta("micro")),
                str(current_version.next_beta("major")),
            ]
        else:
            choices = [
                str(current_version.next_release_candidate("minor")),
                str(current_version.next_release_candidate("micro")),
                str(current_version.next_release_candidate("major")),
            ]
        version = questionary.select(
            f"Which {version} do you want to release?", choices=choices
        ).ask()

    if version:
        return version
    else:
        print("Aborting.")
        sys.exit(1)


def get_rasa_sdk_version() -> Text:
    """Find out what the referenced version of the Rasa SDK is."""
    dependencies_filename = "pyproject.toml"
    toml_data = toml.load(project_root() / dependencies_filename)
    try:
        sdk_version = toml_data["tool"]["poetry"]["dependencies"]["rasa-sdk"]
        if not isinstance(sdk_version, str):
            sdk_version = sdk_version["version"]
        return sdk_version.strip("^ ~")
    except AttributeError:
        raise Exception(f"Failed to find Rasa SDK version in {dependencies_filename}")


def validate_code_is_release_ready(version: Version) -> None:
    """Make sure the code base is valid (e.g. Rasa SDK is up to date)."""
    sdk = Version(get_rasa_sdk_version())
    sdk_version = (sdk.major, sdk.minor)
    rasa_version = (version.major, version.minor)

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
        return "main"


def git_current_branch_is_main_or_release() -> bool:
    """Returns True if the current local git
    branch is main or a release branch e.g. 1.10.x.
    """
    current_branch = git_current_branch()
    return (
        current_branch == "main"
        or RELEASE_BRANCH_PATTERN.match(current_branch) is not None
    )


def create_release_branch(version: Version) -> Text:
    """Create a new branch for this release. Returns the branch name."""
    if version.pre or version.dev:
        branch = f"{PRE_RELEASE_BRANCH_PREFIX}{version}"
    else:
        branch = f"{RELEASE_BRANCH_PREFIX}{version}"

    check_call(["git", "checkout", "-b", branch])
    return branch


def create_commit(version: Version) -> None:
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


def parse_next_version(version: Text) -> Version:
    """Find the next version as a proper semantic version string."""
    if version == "major":
        return Version(get_current_version()).next_major()
    elif version == "minor":
        return Version(get_current_version()).next_minor()
    elif version == "micro":
        return Version(get_current_version()).next_micro()
    elif version == "alpha":
        return Version(get_current_version()).next_alpha()
    elif version == "beta":
        return Version(get_current_version()).next_beta()
    elif version == "rc":
        return Version(get_current_version()).next_release_candidate()
    elif is_valid_version(version):
        return Version(version)
    else:
        raise Exception(f"Invalid version number '{cmdline_args.next_version}'.")


def next_version(args: argparse.Namespace) -> Version:
    """Take cmdline args or ask the user for the next version and return semver."""
    return parse_next_version(args.next_version or ask_version())


def generate_changelog(version: Version) -> None:
    """Call tonwcrier and create a changelog from all available changelog entries."""
    check_call(
        ["towncrier", "build", "--yes", "--version", str(version)],
        cwd=str(project_root()),
    )


def print_done_message(branch: Text, base: Text, version: Version) -> None:
    """Print final information for the user on what to do next."""
    pull_request_url = f"{REPO_BASE_URL}/compare/{base}...{branch}?expand=1"

    print()
    print(f"\033[94m All done - changes for version {version} are ready! \033[0m")
    print()
    print(f"Please open a PR on GitHub: {pull_request_url}")


def print_done_message_same_branch(version: Version) -> None:
    """Print final information for the user in case changes
    are directly committed on this branch.
    """
    print()
    print(
        f"\033[94m All done - changes for version {version} where committed on this branch \033[0m"
    )


def tag_commit(tag: Text) -> None:
    """Tags a git commit."""
    print(f"Applying tag '{tag}' to commit.")
    check_call(["git", "tag", tag, "-m", "next release"])


def push_tag(tag: Text) -> None:
    """Pushes a tag to the remote."""
    print(f"Pushing tag '{tag}' to origin.")
    check_call(["git", "push", "origin", tag, "--tags"])


def print_tag_release_done_message(version: Version) -> None:
    """Print final information for the user about the tagged commit."""
    print()
    print(
        f"\033[94m All done - tag for version {version} "
        "was added and pushed to the remote \033[0m"
    )


def confirm_tag_version(version: Version) -> bool:
    """Allow the user to confirm the version tag they are applying."""
    if str(version) in git_existing_tags():
        confirmed = questionary.confirm(
            f"Tag with version '{version}' already exists, overwrite?", default=False
        ).ask()
    else:
        confirmed = questionary.confirm(
            f"Current version is '{get_current_version()}. "
            f"Is this the tag you want to apply?",
            default=True,
        ).ask()
    if confirmed:
        return True
    else:
        print("Aborting.")
        sys.exit(1)


def prepare_release(args: argparse.Namespace) -> None:
    """Start a release preparation."""
    print(
        "The release script will increase the version number, "
        "create a changelog and create a release branch. Let's go!"
    )

    ensure_clean_git()
    version = next_version(args)

    # Confirm version if interactive mode is enabled or next_version is provided
    if args.interactive or args.next_version:
        confirm_version(version)

    validate_code_is_release_ready(version)

    write_version_file(version)
    write_version_to_pyproject(version)

    if not version.pre and not version.dev:
        # never update changelog on a pre-release version
        generate_changelog(version)

    # alpha or beta workflow on feature branch when a version bump is required
    if (version.is_alpha or version.is_beta) and not git_current_branch_is_main_or_release():
        create_commit(version)
        push_changes()
        print_done_message_same_branch(version)
    else:
        base = git_current_branch()
        branch = create_release_branch(version)
        create_commit(version)
        push_changes()
        print_done_message(branch, base, version)

def tag_release(args: argparse.Namespace) -> None:
    """Tag the current commit with the current version."""
    print(
        """
    The release tag script will tag the current commit with the current version.

    This should be done on the applicable *.x branch after running
    `make prepare-release` and merging the prepared release branch.
        """
    )

    branch = git_current_branch()
    version = Version(get_current_version())

    if (
        not version.is_alpha
        and not version.is_beta
        and not git_current_branch_is_main_or_release()
    ):
        print(
            f"""
    You are currently on branch {branch}.
    You should only apply release tags to release branches (e.g. 1.x) or main.
            """
        )
        sys.exit(1)
    ensure_clean_git()
    if args.skip_confirmation:
        if str(version) in git_existing_tags():
            print(f"Tag with version '{version}' already exists, will not overwrite.")
            sys.exit(1)
    else:
        confirm_tag_version(version)
    tag = str(version)
    tag_commit(tag)
    push_tag(tag)

    print_tag_release_done_message(version)


if __name__ == "__main__":
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    if not hasattr(cmdline_args, "func"):
        arg_parser.print_usage()
        exit()
    cmdline_args.func(cmdline_args)
