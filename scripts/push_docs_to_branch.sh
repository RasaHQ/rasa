#!/bin/bash

set -Eeuo pipefail

TODAY=`date "+%Y%m%d"`
# we build new versions only for minors and majors
PATTERN_FOR_NEW_VERSION="^refs/tags/[0-9]+\\.[0-9]+\\.0$"
PATTERN_FOR_PATCH_VERSION="^refs/tags/[0-9]+\\.[0-9]+\\.[1-9]+$"
MASTER_REF=refs/heads/master
VARIABLES_JSON=docs/docs/variables.json
SOURCES_FILES=docs/docs/sources/
REFERENCE_FILES=docs/docs/reference/
CHANGELOG=docs/docs/changelog.mdx
TELEMETRY_REFERENCE=docs/docs/telemetry/reference.mdx

[[ ! $GITHUB_REF =~ $PATTERN_FOR_NEW_VERSION ]] \
&& [[ ! $GITHUB_REF =~ $PATTERN_FOR_PATCH_VERSION ]] \
&& [[ $GITHUB_REF != $MASTER_REF ]] \
&& echo "Not on master or tagged version, skipping." \
&& exit 0

NEW_VERSION=
EXISTING_VERSION=
if [[ "$GITHUB_REF" =~ $PATTERN_FOR_NEW_VERSION ]]
then
    NEW_VERSION=$(echo $GITHUB_REF | sed -E "s/^refs\/tags\/([0-9]+)\.([0-9]+)\.0$/\1.\2.x/")
elif [[ "$GITHUB_REF" =~ $PATTERN_FOR_PATCH_VERSION ]]
then
    EXISTING_VERSION=$(echo $GITHUB_REF | sed -E "s/^refs\/tags\/([0-9]+)\.([0-9]+)\.[0-9]+$/\1.\2.x/")
fi

# clone the $DOCS_BRANCH in a temp directory
git clone --depth=1 --branch=$DOCS_BRANCH git@github.com:$GITHUB_REPOSITORY.git $TMP_DOCS_FOLDER

if [ ! -z "$NEW_VERSION" ] && [ -d "$TMP_DOCS_FOLDER/docs/versioned_docs/version-$NEW_VERSION/" ]
then
    echo "Trying to create a new docs version, but the folder already exist. Updating the $NEW_VERSION version instead..."
    EXISTING_VERSION=$NEW_VERSION
fi

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
    cd $TMP_DOCS_FOLDER/docs/ && find . ! -path . ! -regex '.*version.*' -exec rm -rf {} + && cd -
    # update the docs/ folder with the latest version of the docs
    cp -R `ls -A | grep -v "^\.git$"` $TMP_DOCS_FOLDER/

    if [ ! -z "$NEW_VERSION" ]
    then
        echo "Generating docs for new version $NEW_VERSION..."
        cd $TMP_DOCS_FOLDER/docs && yarn run new-version $NEW_VERSION && cd -
    fi
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
