#!/bin/bash

set -e

function print_help {
    echo "Available options:"
    echo " start commands (Rasa Core cmdline arguments)  - Start Rasa Core server"
    echo " train                                     - Train a dialogue model"
    echo " start -h                                  - Print Rasa Core help"
    echo " help                                      - Print this help"
    echo " run                                       - Run an arbitrary command inside the container"
}

function download_package {
    case $1 in
        mitie)
            echo "Downloading mitie model..."
            python -m rasa_nlu.download -p mitie
            ;;
        spacy)
            case $2 in 
                en|de)
                    echo "Downloading spacy.$2 model..."
                    python -m spacy download "$2"
                    echo "Done."
                    ;;
                *) 
                    echo "Error. Rasa_nlu supports only english and german models for the time being"
                    print_help
                    exit 1
                    ;;
            esac
            ;;
        *) 
            echo "Error: invalid package specified."
            echo 
            print_help
            ;;
    esac
}

case ${1} in
    start)
        exec python -m rasa_core.server "${@:2}"
        ;;
    run)
        exec "${@:2}"
        ;;
    train)
        exec python -m rasa_core.train "${@:2}"
        ;;
    *)
        print_help
        ;;
esac


