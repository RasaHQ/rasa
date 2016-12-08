#!/bin/bash

set -e

case ${1} in
    start)
        python -m rasa_nlu.server "${@:2}" 
        ;;
    run)
        "${@:2}"
        ;;
    *)
        echo "Available options:"
        echo " start <rasa_nlu_arguments>  - Start RasaNLU server"
        echo " start -h                    - Print RasaNLU help"
        echo " help                        - Print this help" 
        echo " run                         - Run an arbitrary command inside the container"
        ;;
esac

