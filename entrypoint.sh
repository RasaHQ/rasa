#!/bin/bash

set -e

function print_help {
    echo "Available options:"
    echo " start commands (Rasa Core cmdline arguments) - Start Rasa Core server"
    echo " train                                        - Train a dialogue model"
    echo " start -h                                     - Print Rasa Core help"
    echo " help                                         - Print this help"
    echo " run                                          - Run an arbitrary command inside the container"
}

if [[ -v RASA_ACTION_URL ]]; then
    echo "action_endpoint:" > endpoints.yml
    echo "  url: '${RASA_ACTION_URL}'" >> endpoints.yml
fi

if [[ -v RASA_NLU_HOST ]]; then
    echo "nlu:" >> endpoints.yml
    echo "  url: '${RASA_NLU_HOST}'" >> endpoints.yml
fi

case ${1} in
    start)
        exec python -m rasa_core.run --enable_api "${@:2}"
        ;;
    run)
        exec "${@:2}"
        ;;
    train)
        exec python -m rasa_core.train -s ./stories.md -d ./domain.yml -o ./out "${@:2}"
        ;;
    *)
        print_help
        ;;
esac


