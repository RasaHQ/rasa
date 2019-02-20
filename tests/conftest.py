from typing import Text
import os
import pytest

from rasa.cli.constants import (DEFAULT_DOMAIN_PATH, DEFAULT_CONFIG_PATH,
                                DEFAULT_NLU_DATA_PATH, DEFAULT_STORIES_PATH)
from rasa.model import DEFAULT_MODELS_PATH


@pytest.fixture(scope="session")
def project() -> Text:
    import tempfile
    from rasa.cli.scaffold import _create_initial_project

    directory = tempfile.mkdtemp()
    _create_initial_project(directory)

    return directory


def train_model(project: Text, filename: Text = "test.tar.gz"):
    from rasa.cli.train import train
    arguments = type('', (), {})()

    arguments.out = os.path.join(project, DEFAULT_MODELS_PATH, filename)
    arguments.domain = os.path.join(project, DEFAULT_DOMAIN_PATH)
    arguments.config = os.path.join(project, DEFAULT_CONFIG_PATH)
    arguments.nlu = os.path.join(project, DEFAULT_NLU_DATA_PATH)
    arguments.stories = os.path.join(project, DEFAULT_STORIES_PATH)

    train(arguments)

    return arguments.out


@pytest.fixture(scope="session")
def trained_model(project) -> Text:
    return train_model(project)

