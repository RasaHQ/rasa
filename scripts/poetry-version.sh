#!/usr/bin/env python

"""Extract the poetry version from .github/poetry_version.txt. Used in e.g. github workflows."""

import pathlib
import sys
import re

# The poetry version is stored in the .github/poetry_version.txt file due to the https://github.com/python-poetry/poetry/issues/3316
poetry_version_txt = pathlib.Path(__file__).parent.parent / ".github" / "poetry_version.txt"

if __name__ == "__main__":
    version_regex = r'poetry-version=(.*)'
    with poetry_version_txt.open() as f:
        for line in f:
            m = re.search(version_regex, line)
            if m:
                print(m.group(1))
                sys.exit(0)
        else:
            print("Failed to find poetry version.")
            sys.exit(1)
