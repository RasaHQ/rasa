#!/bin/bash

if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ] \
    && [ "$TRAVIS_BRANCH" = "master" ] \
    && [ -n "$TRAVIS_TAG" ]; then

    DIR=docker_minimal
    mkdir $DIR
    cp -r rasa_nlu _pytest entrypoint.sh requirements.txt setup.py config_defaults.json $DIR
    cp Dockerfile_minimal $DIR/Dockerfile
    cd $DIR
    docker build -t golastmile/rasa_nlu_minimal:$TRAVIS_TAG -t golastmile/rasa_nlu_minimal:latest .
    rm -rf $DIR

    DIR=docker_full
    mkdir $DIR
    cp dev-requirements.txt $DIR/dev-requirements.txt
    cp Dockerfile_full $DIR/Dockerfile
    cd $DIR
    docker build -t golastmile/rasa_nlu:$1 -t golastmile/rasa_nlu:latest .
    rm -rf $DIR

    docker login -u="$DOCKER_USERNAME" -p="$DOCKER_PASSWORD"
    docker push golastmile/rasa_nlu_minimal:"$TRAVIS_TAG"
    docker push golastmile/rasa_nlu_minimal:latest
    docker push golastmile/rasa_nlu:"$TRAVIS_TAG"
    docker push golastmile/rasa_nlu:latest
fi