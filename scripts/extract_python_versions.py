#!/usr/bin/env python3
"""
Extract Python versions from pyproject.toml for EOL checking.

This script parses the Python version constraint from pyproject.toml and generates
a list of major.minor versions that need to be checked for end-of-life status.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def parse_version_constraint(version_string: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse a version constraint string to extract min and max versions.

    Args:
        version_string: Version constraint like ">=3.9.2,<3.12"

    Returns:
        Tuple of (min_version, max_version) as major.minor strings
    """
    # Remove quotes and whitespace
    version_string = version_string.strip().strip('"\'')

    min_version = None
    max_version = None

    # Extract minimum version (e.g., ">=3.9.2" -> "3.9")
    min_match = re.search(r'>=(\d+\.\d+)', version_string)
    if min_match:
        min_version = min_match.group(1)

    # Extract maximum version (e.g., "<3.12" -> "3.12")
    max_match = re.search(r'<(\d+\.\d+)', version_string)
    if max_match:
        max_version = max_match.group(1)

    return min_version, max_version


def generate_version_list(min_version: Optional[str], max_version: Optional[str]) -> List[str]:
    """
    Generate a list of major.minor versions between min and max (exclusive).

    Args:
        min_version: Minimum version (e.g., "3.9")
        max_version: Maximum version (e.g., "3.12")

    Returns:
        List of version strings (e.g., ["3.9", "3.10", "3.11"])
    """
    if not min_version or not max_version:
        return []

    min_major, min_minor = map(int, min_version.split('.'))
    max_major, max_minor = map(int, max_version.split('.'))

    versions = []

    for major in range(min_major, max_major + 1):
        if major == min_major and major == max_major:
            # Same major version, iterate through minors
            for minor in range(min_minor, max_minor):
                versions.append(f"{major}.{minor}")
        elif major == min_major:
            # First major version, start from min_minor
            for minor in range(min_minor, 10):  # Assume max minor is 9
                versions.append(f"{major}.{minor}")
        elif major == max_major:
            # Last major version, go up to max_minor-1
            for minor in range(0, max_minor):
                versions.append(f"{major}.{minor}")
        else:
            # Middle major versions, include all minors
            for minor in range(0, 10):  # Assume max minor is 9
                versions.append(f"{major}.{minor}")

    return versions


def extract_python_versions_from_pyproject(pyproject_path: Path) -> List[str]:
    """
    Extract Python versions from pyproject.toml file.

    Args:
        pyproject_path: Path to pyproject.toml file

    Returns:
        List of Python version strings to check for EOL

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        ValueError: If Python version constraint is not found or invalid
    """
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    with open(pyproject_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Look for python version constraint in [tool.poetry.dependencies] section
    python_pattern = r'^python\s*=\s*["\']([^"\']+)["\']'

    match = re.search(python_pattern, content, re.MULTILINE)
    if match:
        version_constraint = match.group(1)
        min_version, max_version = parse_version_constraint(version_constraint)

        if not min_version or not max_version:
            raise ValueError(f"Could not parse version constraint: {version_constraint}")

        return generate_version_list(min_version, max_version)

    raise ValueError("Python version constraint not found in pyproject.toml")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract Python versions from pyproject.toml for EOL checking"
    )
    parser.add_argument(
        "--pyproject-path",
        type=Path,
        default=Path("pyproject.toml"),
        help="Path to pyproject.toml file (default: pyproject.toml)"
    )
    args = parser.parse_args()

    try:
        versions = extract_python_versions_from_pyproject(args.pyproject_path)
        print(",".join(versions))

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
