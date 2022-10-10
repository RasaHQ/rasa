#!/bin/bash

# In case this script fails in our CI workflows, you can run it locally.
# For instance, if the doc version for 2.2.10 fails to be pushed, you
# can do:
#
# 1. make install install-docs
# 2. make prepare-docs
# 3. TMP_DOCS_FOLDER=/tmp/documentation DOCS_BRANCH=documentation GITHUB_REF=refs/tags/2.2.10 GITHUB_REPOSITORY=RasaHQ/rasa ./scripts/push_docs_to_branch.sh
#
# This script will push the changes to the `documentation` branch of this repository.

set -Eeuo pipefail

TODAY=`date "+%Y%m%d"`
# we build new versions only for minors and majors
PATTERN_FOR_NEW_VERSION="^refs/tags/[0-9]+\\.0\\.0$"
PATTERN_FOR_EXISTING_VERSION="^refs/tags/[0-9]+\\.[0-9]+\\.[0-9]+$"
MAIN_REF=refs/heads/main
VARIABLES_JSON=docs/docs/variables.json
SOURCES_FILES=docs/docs/sources/
REFERENCE_FILES=docs/docs/reference/
CHANGELOG=docs/docs/changelog.mdx
TELEMETRY_REFERENCE=docs/docs/telemetry/reference.mdx

[[ ! $GITHUB_REF =~ $PATTERN_FOR_NEW_VERSION ]] \
&& [[ ! $GITHUB_REF =~ $PATTERN_FOR_EXISTING_VERSION ]] \
&& [[ $GITHUB_REF != $MAIN_REF ]] \
&& echo "Not on main or tagged version, skipping." \
&& exit 0

NEW_VERSION=
EXISTING_VERSION=
if [[ "$GITHUB_REF" =~ $PATTERN_FOR_NEW_VERSION ]]
then
    NEW_VERSION=$(echo $GITHUB_REF | sed -E "s/^refs\/tags\/([0-9]+)\.([0-9]+)\.0$/\1.x/")
    if [[ -n ${CI} ]]; then echo "New version: ${NEW_VERSION}"; fi
elif [[ "$GITHUB_REF" =~ $PATTERN_FOR_EXISTING_VERSION ]]
then
    EXISTING_VERSION=$(echo $GITHUB_REF | sed -E "s/^refs\/tags\/([0-9]+)\.([0-9]+)\.[0-9]+$/\1.x/")
    if [[ -n ${CI} ]]; then echo "Existing version: ${EXISTING_VERSION}"; fi
fi

# clone the $DOCS_BRANCH in a temp directory
git clone --depth=1 --branch=$DOCS_BRANCH git@github.com:$GITHUB_REPOSITORY.git $TMP_DOCS_FOLDER

if [ ! -z "$NEW_VERSION" ] && [ -d "$TMP_DOCS_FOLDER/docs/versioned_docs/version-$NEW_VERSION/" ]
then
    echo "Trying to create a new docs version, but the folder already exist. Updating the $NEW_VERSION version instead..."
    EXISTING_VERSION=$NEW_VERSION
fi

# install yarn dependencies in the temp directory
cd $TMP_DOCS_FOLDER/docs && yarn install && cd - || exit 1

if [ ! -z "$EXISTING_VERSION" ]
then
    echo "Updating docs for existing version $EXISTING_VERSION..."
    # FIXME: this doesn't support all types of docs updates on an existing version at the moment,
    # For instance if we were to make significant updates to the documentation pages
    # (creating new page, deleting some, updating the sidebar), these changes wouldn't work here.
    cp -R docs/docs/* $TMP_DOCS_FOLDER/docs/versioned_docs/version-$EXISTING_VERSION/
else
    echo "Updating the docs..."
    # remove everything in the previous docs/ folder, except versioned_docs/*, versioned_sidebars/*
    # and versions.js files.
    # skip node_modules/ because `yarn install` has run
    cd $TMP_DOCS_FOLDER/docs/ && find  . ! -path . ! -regex '.*\(version\|node_modules\).*' -exec rm -rf {} + && cd - || exit 1
    # update the docs/ folder with the latest version of the docs
    cp -R `ls -A | grep -v "^\.git$"` $TMP_DOCS_FOLDER/

    if [ ! -z "$NEW_VERSION" ]
    then
        echo "Generating docs for new version $NEW_VERSION..."
        cd $TMP_DOCS_FOLDER/docs && yarn run new-version $NEW_VERSION && cd - || exit 1
    fi
fi

CURRENTLY_EDITING_VERSION=${EXISTING_VERSION:-$NEW_VERSION}
if [ -n "$CURRENTLY_EDITING_VERSION" ]
then
    cd $TMP_DOCS_FOLDER/docs && yarn run update-versioned-sources $CURRENTLY_EDITING_VERSION && cd - || exit 1
fi

cd $TMP_DOCS_FOLDER

if [ -z "$(git status --porcelain)" ]
then
    echo "Nothing changed in docs, done 👍"
else
    echo "Pushing changes to git..."
    git add .
    git add --force $VARIABLES_JSON $SOURCES_FILES $CHANGELOG $REFERENCE_FILES $TELEMETRY_REFERENCE
    git commit -am "AUTO docusaurus $TODAY"
    git fetch --unshallow
    git push origin $DOCS_BRANCH

    echo "Done 👌"
fi
