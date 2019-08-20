#!/bin/bash

# to enable remote execution
cd "$(dirname "$0")"
cd ..

virtualenv -p /usr/bin/python3 env

source env/bin/activate
pip3 install --upgrade pip
python3 --version
pip3 --version
pip3 install -r ./requirements.txt
pip3 install -r ./requirements-dev.txt

make prepare-tests-ubuntu
