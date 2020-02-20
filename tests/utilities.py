from typing import Text
from yarl import URL

import rasa.utils.io as io_utils
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.nlu.selectors.diet_selector import DIETSelector
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

        if component["name"] in [
            EmbeddingIntentClassifier.name,
            DIETClassifier.name,
            ResponseSelector.name,
            DIETSelector.name,
        ]:
            component[EPOCHS] = 2

    io_utils.write_yaml_file(config, output_file)
