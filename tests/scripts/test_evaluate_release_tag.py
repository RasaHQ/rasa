from scripts.evaluate_release_tag import filter_ga_relases, should_build_docs
import pytest
from pep440_version_utils import Version
from typing import List
from unittest.mock import patch


@pytest.mark.parametrize(
    "releases, expected",
    [
        (
            [Version("1.1.0"), Version("2.2.0")],
            [Version("1.1.0"), Version("2.2.0")],
        ),
        (
            [Version("1.1.0"), Version("2.2.0"), Version("1.1.1a")],
            [Version("1.1.0"), Version("2.2.0")],
        ),
        (
            [
                Version("1.1.0"),
                Version("2.2.0"),
                Version("1.1.1a1"),
                Version("1.1.1b1"),
                Version("1.1.1rc1"),
            ],
            [Version("1.1.0"), Version("2.2.0")],
        ),
    ],
)
def test_filter_ga_releases(releases: List[Version], expected: List[Version]):
    result = filter_ga_relases(releases)
    assert result == expected


@pytest.mark.parametrize(
    "releases, tag, expected",
    [
        (
            [Version("1.1.0"), Version("2.2.0")],
            Version("2.3.0"),
            True,
        ),
        (
            [Version("1.1.0"), Version("2.2.0"), Version("2.3.0a1")],
            Version("2.2.1"),
            True,
        ),
        (
            [
                Version("1.1.0"),
                Version("2.2.0"),
                Version("2.3.0b1"),
                Version("2.4.0a1"),
            ],
            Version("2.2.1"),
            True,
        ),
        (
            [Version("1.1.0"), Version("2.2.0"), Version("2.3.0")],
            Version("1.2.0"),
            False,
        ),
        (
            [Version("1.1.0"), Version("1.2.0a1"), Version("2.3.0")],
            Version("1.1.2"),
            False,
        ),
        (
            [Version("1.1.0"), Version("2.2.0"), Version("2.3.0")],
            Version("2.2.1"),
            False,
        ),
        (
            [Version("1.1.0"), Version("2.2.0"), Version("2.3.0")],
            Version("2.2.1a1"),
            False,
        ),
    ],
)
@patch("scripts.evaluate_release_tag.git_existing_tag_versions")
def test_should_build_docs(
    mock_get_existing_tag_versions,
    releases: List[Version],
    tag: Version,
    expected: bool,
) -> None:
    mock_get_existing_tag_versions.return_value = releases
    result = should_build_docs(tag)
    assert result == expected
