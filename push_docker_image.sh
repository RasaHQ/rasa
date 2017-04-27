#!/bin/bash

if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ] && [ "$TRAVIS_PULL_REQUEST" = "false" ]; then
    KEYFILE=golastmile-9baf26f3fde2.json
    openssl aes-256-cbc -K $encrypted_faee90f6e198_key -iv $encrypted_faee90f6e198_iv -in golastmile-9baf26f3fde2.json.enc -out "$KEYFILE" -d

    if [ ! -d "$HOME/google-cloud-sdk/bin" ]; then rm -rf $HOME/google-cloud-sdk; curl https://sdk.cloud.google.com | bash; fi
    source "$HOME/google-cloud-sdk/path.bash.inc"
    gcloud auth activate-service-account "$GCLOUD_EMAIL" --key-file "$KEYFILE"

    docker build \
        -t gcr.io/golastmile/rasa_nlu:"$TRAVIS_BRANCH" \
        -t gcr.io/golastmile/rasa_nlu:"$TRAVIS_BRANCH"_"$TRAVIS_BUILD_NUMBER" .

    gcloud docker -- push gcr.io/golastmile/rasa_nlu:"$TRAVIS_BRANCH"
    gcloud docker -- push gcr.io/golastmile/rasa_nlu:"$TRAVIS_BRANCH"_"$TRAVIS_BUILD_NUMBER"

    if [ -n "$TRAVIS_TAG" ]; then
        docker tag gcr.io/golastmile/rasa_nlu:"$TRAVIS_BRANCH" gcr.io/golastmile/rasa_nlu:"$TRAVIS_TAG"
        gcloud docker -- push gcr.io/golastmile/rasa_nlu:"$TRAVIS_TAG"
    fi
fi