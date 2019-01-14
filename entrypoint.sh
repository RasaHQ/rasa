#!/bin/bash

set -e

function print_help {
    echo "Available options:"
    echo " start commands (rasa cmd line arguments)  - Start RasaNLU server"
    echo " download {mitie, spacy en, spacy de}      - Download packages for mitie or spacy (english or german)"
    echo " start -h                                  - Print RasaNLU help"
    echo " help                                      - Print this help"
    echo " run                                       - Run an arbitrary command inside the container"
}

function download_package {
    case $1 in
        mitie)
            echo "Downloading mitie model..."
            wget https://github.com/mit-nlp/MITIE/releases/download/v0.4/MITIE-models-v0.2.tar.bz2
            tar jxf MITIE-models-v0.2.tar.bz2
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
        exec python -m rasa_nlu.server "${@:2}" 
        ;;
    run)
        exec "${@:2}"
        ;;
    download)
        download_package ${@:2}
        ;;
    *)
        print_help
        ;;
esac


