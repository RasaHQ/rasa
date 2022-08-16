"""Evaluate release tag for whether docs should be built or not.

"""
import os
import argparse
from pathlib import Path
from subprocess import check_output
from typing import List, Text, Set

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


def git_existing_tag_versions() -> Set[Text]:
    """Return all existing tags in the local git repo."""

    stdout = check_output(["git", "tag"])
    tags = set(stdout.decode().split("\n"))
    versions = [Version(tag) for tag in tags if is_valid_version(tag)]
    return versions

def git_plain_tag_versions(versions: List[Version]) -> Set[Text]:
    """Return non-alpha/rc existing tags"""
    return [version for version in versions if is_plain_version(version)]


def main(args: argparse.Namespace) -> None:
    tag = Version(args.tag)

    existing_tags = git_existing_tag_versions()
    existing_tags.sort()
    latest_version = existing_tags[-1]
    print(f"The latest non-alpha/rc/nightly tag is {latest_version}.")

    previous_major = latest_version.major - 1
    previous_major_versions = [tag for tag in existing_tags if tag.major == previous_major]
    previous_major_latest_version = previous_major_versions[-1]
    print(f"The latest tag on the previous major is {previous_major_latest_version}.")

    build_docs = False

    if not is_plain_version(tag):
        print(f"Tag {tag} is an alpha, rc, nightly, or otherwise non-standard version.")
    elif tag > latest_version:
        print(f"Tag {tag} is higher than the latest version. Docs should be built.")
        build_docs = True
    elif tag.major == previous_major and tag > previous_major_latest_version:
        print(f"Tag {tag} is higher than the latest version for the previous major. Docs should be built.")
        build_docs = True

    if not build_docs:
        raise SystemExit(f"Docs should not be built for tag {tag}.")


if __name__ == "__main__":
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()
    main(cmdline_args)

