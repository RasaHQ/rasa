#!/bin/bash

set -Eeuo pipefail

function print_help {
    echo "Available options:"
    echo " start commands (Rasa Core cmdline arguments) - Start Rasa Core server"
    echo " train                                        - Train a dialogue model"
    echo " start -h                                     - Print Rasa Core help"
    echo " help                                         - Print this help"
    echo " run                                          - Run an arbitrary command inside the container"
}

case ${1} in
    start)
        exec python -m rasa_core.run --enable_api "${@:2}"
        ;;
    run)
        exec "${@:2}"
        ;;
    train)
        exec python -m rasa_core.train -s project/stories.md -d project/domain.yml -o ./model "${@:2}"
        ;;
    *)
        print_help
        ;;
esac


