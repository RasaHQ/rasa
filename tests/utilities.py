from yarl import URL
from typing import Text

import rasa.utils.io as io_utils

from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.utils.tensorflow.constants import EPOCHS


def latest_request(mocked, request_type, path):
    return mocked.requests.get((request_type, URL(path)))


def json_of_latest_request(r):
    return r[-1].kwargs["json"]


def update_number_of_epochs(config_path: Text, output_file: Text):
    config = io_utils.read_yaml_file(config_path)

    if "pipeline" not in config.keys():
        raise ValueError(f"Invalid config provided! File: '{config_path}'.")

    for component in config["pipeline"]:
        # do not update epochs for pipeline templates
        if not isinstance(component, dict):
            continue

        if component["name"] in [DIETClassifier.name, ResponseSelector.name]:
            component[EPOCHS] = 1

    io_utils.write_yaml(config, output_file)
