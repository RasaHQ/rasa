#!/bin/bash

set -euo pipefail
# Lint docstrings only against the the diff to avoid too many errors.
# Check only production code. Ignore other other errors which are captured by `lint`

# Compare against `main` if no branch was provided
BRANCH="${1:-main}"
# Diff of committed changes (shows only changes introduced by your branch
FILES_WITH_DIFF=`git diff $BRANCH...HEAD --name-only -- rasa/**/*.py | xargs echo -n`

if [ ! -z "$FILES_WITH_DIFF" ]
then
    poetry run ruff check --select D --diff $FILES_WITH_DIFF
else
    echo "No python files in diff."
fi

# Diff of uncommitted changes for running locally
DEV_FILES_WITH_DIFF=`git diff HEAD --name-only -- rasa | xargs echo -n`

if [ ! -z "$DEV_FILES_WITH_DIFF" ]
then
    poetry run ruff check --select D --diff $DEV_FILES_WITH_DIFF
fi
