#!/usr/bin/env python3
"""Compare top-level dependencies between two git refs by parsing pyproject.toml.

This script extracts dependencies directly from pyproject.toml without requiring
any package installation, making it fast and lightweight for CI usage.

Usage:
    python find_new_dependencies.py <base_ref> <head_ref>
    python find_new_dependencies.py origin/main HEAD

Legacy usage (for backwards compatibility):
    MAIN_OUTPUT=dep1,dep2 PR_OUTPUT=dep1,dep2,dep3 python find_new_dependencies.py
"""
from __future__ import annotations

import os
import subprocess
import sys

# tomllib is built-in for Python 3.11+, use tomli as fallback for older versions
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


def get_file_from_git_ref(ref: str, filepath: str) -> str:
    """Get file contents from a specific git ref."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{filepath}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def extract_dependencies_from_pyproject(content: str) -> set[str]:
    """Extract top-level dependency names from pyproject.toml content."""
    data = tomllib.loads(content)

    dependencies = set()

    # Get main dependencies
    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for dep_name in poetry_deps:
        if dep_name.lower() != "python":  # Exclude python version constraint
            dependencies.add(dep_name.lower())

    # Get dependencies from poetry groups (dev, test, etc.)
    groups = data.get("tool", {}).get("poetry", {}).get("group", {})
    for group_name, group_data in groups.items():
        group_deps = group_data.get("dependencies", {})
        for dep_name in group_deps:
            dependencies.add(dep_name.lower())

    return dependencies


def find_new_dependencies_from_refs(base_ref: str, head_ref: str) -> set[str]:
    """Find new dependencies by comparing pyproject.toml between two git refs."""
    base_pyproject = get_file_from_git_ref(base_ref, "pyproject.toml")
    head_pyproject = get_file_from_git_ref(head_ref, "pyproject.toml")

    base_deps = extract_dependencies_from_pyproject(base_pyproject)
    head_deps = extract_dependencies_from_pyproject(head_pyproject)

    return head_deps - base_deps


def find_new_dependencies_from_env() -> set[str]:
    """Legacy: Find new dependencies from environment variables."""
    main_output = os.environ.get("MAIN_OUTPUT", "")
    pr_output = os.environ.get("PR_OUTPUT", "")

    main_dependencies = set(d.strip().lower() for d in main_output.split(",") if d.strip())
    pr_dependencies = set(d.strip().lower() for d in pr_output.split(",") if d.strip())

    return pr_dependencies - main_dependencies


def main() -> None:
    if len(sys.argv) >= 3:
        # New usage: compare git refs
        base_ref = sys.argv[1]
        head_ref = sys.argv[2]
        new_deps = find_new_dependencies_from_refs(base_ref, head_ref)
    else:
        # Legacy usage: environment variables
        new_deps = find_new_dependencies_from_env()

    if new_deps:
        print(", ".join(sorted(new_deps)))


if __name__ == "__main__":
    main()
