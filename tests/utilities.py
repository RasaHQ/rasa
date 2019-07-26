import os
import time
from typing import Optional, Text

from yarl import URL

from constants import DEFAULT_CONFIG_PATH, DEFAULT_DATA_PATH
from importers.rasa import RasaFileImporter
from model import (
    FINGERPRINT_CONFIG_KEY,
    FINGERPRINT_CONFIG_CORE_KEY,
    FINGERPRINT_CONFIG_NLU_KEY,
    FINGERPRINT_DOMAIN_KEY,
    FINGERPRINT_TRAINED_AT_KEY,
    FINGERPRINT_RASA_VERSION_KEY,
    FINGERPRINT_STORIES_KEY,
    FINGERPRINT_NLU_DATA_KEY,
)


def latest_request(mocked, request_type, path):
    return mocked.requests.get((request_type, URL(path)))


def json_of_latest_request(r):
    return r[-1].kwargs["json"]


def _fingerprint(
    config: Optional[Text] = None,
    config_nlu: Optional[Text] = None,
    config_core: Optional[Text] = None,
    domain: Optional[int] = None,
    rasa_version: Text = "1.0",
    stories: Optional[int] = None,
    nlu: Optional[int] = None,
):
    return {
        FINGERPRINT_CONFIG_KEY: config if config is not None else ["test"],
        FINGERPRINT_CONFIG_CORE_KEY: config_core
        if config_core is not None
        else ["test"],
        FINGERPRINT_CONFIG_NLU_KEY: config_nlu if config_nlu is not None else ["test"],
        FINGERPRINT_DOMAIN_KEY: domain if domain is not None else ["test"],
        FINGERPRINT_TRAINED_AT_KEY: time.time(),
        FINGERPRINT_RASA_VERSION_KEY: rasa_version,
        FINGERPRINT_STORIES_KEY: stories if stories is not None else ["test"],
        FINGERPRINT_NLU_DATA_KEY: nlu if nlu is not None else ["test"],
    }


def _project_files(
    project,
    config_file=DEFAULT_CONFIG_PATH,
    domain="domain.yml",
    training_files=DEFAULT_DATA_PATH,
):
    paths = {
        "config_file": config_file,
        "domain_path": domain,
        "training_data_paths": training_files,
    }

    paths = {k: v if v is None else os.path.join(project, v) for k, v in paths.items()}
    paths["training_data_paths"] = [paths["training_data_paths"]]

    return RasaFileImporter(**paths)


class ResponseTest(object):
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload
