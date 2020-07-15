#!/bin/bash

set -Eeuo pipefail

TMP_DOCS_FOLDER=tmp-documentation
TODAY=`date "+%Y%m%d"`

# use a temp directory, clone the $DOCS_BRANCH in it
echo "Cloning $GITHUB_REPOSITORY#$DOCS_BRANCH into $TMP_DOCS_FOLDER..."
mkdir $TMP_DOCS_FOLDER
git clone --depth=1 --branch $DOCS_BRANCH git@github.com:$GITHUB_REPOSITORY.git $TMP_DOCS_FOLDER

echo "Updating the docs..."
cp $DOCS_FOLDER/* $TMP_DOCS_FOLDER/
cd $TMP_DOCS_FOLDER

if [ -z "$NEW_VERSION" ]
then
    echo "Skipping generation of new version."
else
    echo "Generating docs for new version $NEW_VERSION..."
    yarn run new-version $NEW_VERSION
fi

echo "Pushing changes to git..."
git add .
git ci -am "AUTO docusaurus $TODAY"
git push origin $DOCS_BRANCH

echo "Done 👌"
