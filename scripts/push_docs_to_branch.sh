#!/bin/bash

set -Eeuo pipefail

RASA_FOLDER=rasa
TESTS_FOLDER=tests
DATA_FOLDER=data
TMP_DOCS_FOLDER=tmp-documentation
TODAY=`date "+%Y%m%d"`
# we build new versions only for majors
PATTERN_FOR_NEW_VERSION="^refs/tags/[0-9]+\\.0\\.0$"
MASTER_REF=refs/heads/new-docs-ci-build

[[ ! $GITHUB_REF =~ $PATTERN_FOR_NEW_VERSION ]] \
&& [[ $GITHUB_REF != $MASTER_REF ]] \
&& echo "Not on master or major version, skipping." \
&& exit 0

NEW_VERSION=
if [ "$GITHUB_REF" != $MASTER_REF ]
then
    NEW_VERSION=${GITHUB_REF/refs\/tags\//}
fi

# clone the $DOCS_BRANCH in a temp directory
git clone --depth=1 --branch=$DOCS_BRANCH git@github.com:$GITHUB_REPOSITORY.git $TMP_DOCS_FOLDER

echo "Updating the docs..."
cp -R $DOCS_FOLDER $TMP_DOCS_FOLDER/docs
# we need these because they are imported by docs files
cp -R $RASA_FOLDER $TMP_DOCS_FOLDER/$RASA_FOLDER
cp -R $DATA_FOLDER $TMP_DOCS_FOLDER/$DATA_FOLDER
cp -R $TESTS_FOLDER $TMP_DOCS_FOLDER/$TESTS_FOLDER

cd $TMP_DOCS_FOLDER

if [ ! -z "$NEW_VERSION" ]
then
    echo "Generating docs for new version $NEW_VERSION..."
    yarn run new-version $NEW_VERSION
fi

if [ -z "$(git status --porcelain)" ]
then

    echo "Pushing changes to git..."
    git add .
    git ci -am "AUTO docusaurus $TODAY"
    git push origin $DOCS_BRANCH

    echo "Done 👌"
else
    echo "Nothing changed in docs, done 👍"
fi
