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
            python -m rasa_nlu.download -p mitie
            ;;
        spacy)
            case $2 in 
                en|de)
                    echo "Downloading spacy.$2 model..."
                    python -m spacy."$2".download all
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

# If there is only a model, set it as default model
function set_default_model {
   if [ -d "/app/models" ]
     echo "Models directory exists"
     then
     if [ ! -d "/app/models/default" ]
       then
       echo "Default model doesn't exist"
       if [ "x1" == x`ls -ld /app/models/* | wc -l` ]
          then
          echo "Renaming existing model to default"
          mv `ls -d /app/models/*` /app/models/default
          if [ -d "/app/models/default" ]
          then
              echo "Model renamed to default"
          else
              echo "Model couldn't be renamed to default"
          fi
       else
          echo "There are no models or more than one"
       fi
     fi
   fi
}



case ${1} in
    start)
        set_default_model
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


