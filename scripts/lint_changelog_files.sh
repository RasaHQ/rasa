#!/bin/bash

set -euo pipefail
# Lint changelog filenames to avoid merging of incorrectly named changelog fragment files
# For more info about proper changelog file naming, see https://github.com/RasaHQ/rasa/blob/main/changelog/README.md
FILES=`ls -A changelog/ | grep -v -E "^([0-9]+.(misc|doc|improvement|feature|removal|bugfix).md|README.md|_template.md.jinja2)$" || true`
if [ -z "$FILES" ]
then
    echo "Changelog files are properly configured."
else
    echo "Unexpected files under the changelog/ folder:"
    echo $FILES
    exit 1
fi
