#!/bin/bash

set -Eeuo pipefail

function print_help {
    echo "Available options:"
    echo " start commands (Rasa cmdline arguments) - Start Rasa server"
    echo " download {mitie, spacy en, spacy de}    - Download packages for mitie or spacy (english or german)"
    echo " train                                   - Train a model"
    echo " start -h                                - Print Rasa help"
    echo " help                                    - Print this help"
    echo " run                                     - Run an arbitrary command inside the container"
}

case ${1} in
    start)
        exec python3 -m rasa.core.run --enable_api "${@:2}"
        ;;
    run)
        exec "${@:2}"
        ;;
    train)
        exec python3 -m rasa.core.train -s project/stories.md -d project/domain.yml -o ./model "${@:2}"
        ;;
    download)
        download_package ${@:2}
        ;;
    *)
        print_help
        ;;
esac
