"""Evaluate release tag for whether docs should be built or not.

"""
import argparse
from subprocess import check_output
from typing import List

from pep440_version_utils import Version, is_valid_version


def create_argument_parser() -> argparse.ArgumentParser:
    """Parse all the command line arguments for the release script."""

    parser = argparse.ArgumentParser(description="Evaluate whether docs should be built for release tag.")
    parser.add_argument(
        "tag",
        type=str,
        help="The tag currently being released.",
    )

    return parser


def is_plain_version(version: Version) -> bool:
    """Check whether a version is a X.x.x release with no alpha or rc markers"""
    return version.base_version == str(version)


def git_existing_tag_versions() -> List[Version]:
    """Return all existing tags in the local git repo."""
    stdout = check_output(["git", "tag"])
    tags = set(stdout.decode().split("\n"))
    versions = [Version(tag) for tag in tags if is_valid_version(tag)]
    return versions


def git_plain_tag_versions(versions: List[Version]) -> List[Version]:
    """Return non-alpha/rc existing tags"""
    return [version for version in versions if is_plain_version(version)]

def filter_ga_relases(tags: List[Version]) -> List[Version]:
    """Return all general availability (GA) releases"""
    return [tag for tag in tags if tag.is_alpha is False and tag.is_beta is False and tag.is_release_candidate is False]


def should_build_docs(tag: Version) -> bool:
    """Docs should be built only for the latest GA release tags"""
    existing_tags = git_existing_tag_versions()

    ga_releases = filter_ga_relases(existing_tags)
    ga_releases.sort()
    latest_version = ga_releases[-1]
    print(f"The latest General Availability release tag is {latest_version}.")
    need_to_build_docs = False

    if not is_plain_version(tag):
        print(f"Tag {tag} is an alpha, beta, rc, nightly, or otherwise non-standard version.")
    elif tag >= latest_version:
        print(f"Tag {tag} is the latest version. Docs should be built.")
        need_to_build_docs = True

    return need_to_build_docs


def main(args: argparse.Namespace) -> None:
    tag = Version(args.tag)
    if not should_build_docs(tag):
        raise SystemExit(f"Docs should not be built for tag {tag}.")


if __name__ == "__main__":
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    main(cmdline_args)
