#!/bin/bash

# to enable remote execution
cd "$(dirname "$0")"

source ./env/bin/activate

pip3 install -e .
