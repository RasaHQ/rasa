#!/usr/bin/env python

"""Extract the poetry version from pyproject.toml. Used in e.g. github workflows."""

import sys
import re

if __name__ == "__main__":
    version_regex = r'"poetry[^\"]*=([^\"]+)"'
    with open("pyproject.toml") as f:
        for line in f:
            m = re.search(version_regex, line)
            if m:
                print(m.group(1))
                sys.exit(0)
        else:
            print("Failed to find poetry version.")
            sys.exit(1)
