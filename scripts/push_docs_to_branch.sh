#!/bin/bash

set -Eeuo pipefail

TODAY=`date "+%Y%m%d"`
# we build new versions only for minors and majors
PATTERN_FOR_NEW_VERSION="^refs/tags/[0-9]+\\.[0-9]+\\.0$"
PATTERN_FOR_PATCH_VERSION="^refs/tags/[0-9]+\\.[0-9]+\\.[1-9]+$"
MASTER_REF=refs/heads/master

[[ ! $GITHUB_REF =~ $PATTERN_FOR_NEW_VERSION ]] \
&& [[ ! $GITHUB_REF =~ $PATTERN_FOR_PATCH_VERSION ]] \
&& [[ $GITHUB_REF != $MASTER_REF ]] \
&& echo "Not on master or tagged version, skipping." \
&& exit 0

NEW_VERSION=
EXISTING_VERSION=
if [[ "$GITHUB_REF" =~ $PATTERN_FOR_NEW_VERSION ]]
then
    NEW_VERSION=${GITHUB_REF/refs\/tags\//}
elif [[ "$GITHUB_REF" =~ $PATTERN_FOR_PATCH_VERSION ]]
then
    EXISTING_VERSION=$(echo $GITHUB_REF | sed -E "s/^refs\/tags\/([0-9]+)\.([0-9]+)\.[0-9]+$/\1.\2.0/")
fi

# clone the $DOCS_BRANCH in a temp directory
git clone --depth=1 --branch=$DOCS_BRANCH git@github.com:$GITHUB_REPOSITORY.git $TMP_DOCS_FOLDER

echo "Updating the docs..."
# FIXME: remove the next 2 lines when we do the move
mv docs olddocs
mv newdocs docs
cp -R `ls -A | grep -v "^\.git$"` $TMP_DOCS_FOLDER/
# FIXME: remove the next 3 lines when we do the move
rm -rf $TMP_DOCS_FOLDER/olddocs
mv docs newdocs
mv olddocs docs


cd $TMP_DOCS_FOLDER

if [ ! -z "$NEW_VERSION" ]
then
    echo "Generating docs for new version $NEW_VERSION..."
    cd docs
    yarn run new-version $NEW_VERSION
    cd ..
fi

if [ ! -z "$EXISTING_VERSION" ]
then
    echo "Updating docs for existing version $EXISTING_VERSION..."
    cd docs
    cp -R docs/ versioned_docs/version-$EXISTING_VERSION/
    # remove updates to the "next" version
    git checkout docs/
    git clean docs/
fi

if [ -z "$(git status --porcelain)" ]
then
    echo "Nothing changed in docs, done 👍"
else
    echo "Pushing changes to git..."
    git add .
    git commit -am "AUTO docusaurus $TODAY"
    git fetch --unshallow
    git push origin $DOCS_BRANCH

    echo "Done 👌"
fi
