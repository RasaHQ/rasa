#!/bin/bash

if [ "$TRAVIS_PYTHON_VERSION" = "2.7" ] \
    && [ "$TRAVIS_BRANCH" = "master" ] \
    && [ -n "$TRAVIS_TAG" ]; then

    DIR=docker_minimal
    mkdir $DIR
    cp -r rasa_nlu _pytest test_models entrypoint.sh requirements.txt setup.py config_defaults.json $DIR
    cp Dockerfile_minimal $DIR/Dockerfile
    cd $DIR
    docker build -t golastmile/rasa_nlu:$TRAVIS_TAG -t golastmile/rasa_nlu:latest .
    cd ..

    DIR=docker_full
    mkdir $DIR
    cp dev-requirements.txt $DIR/dev-requirements.txt
    cp Dockerfile_full $DIR/Dockerfile
    cd $DIR
    docker build -t golastmile/rasa_nlu_full:$TRAVIS_TAG -t golastmile/rasa_nlu_full:latest .
    cd ..

    docker login -u="$DOCKER_USERNAME" -p="$DOCKER_PASSWORD"
    docker push golastmile/rasa_nlu:"$TRAVIS_TAG"
    docker push golastmile/rasa_nlu:latest
    docker push golastmile/rasa_nlu_full:"$TRAVIS_TAG"
    docker push golastmile/rasa_nlu_full:latest
fi