from typing import Text
import os
import pytest

from rasa.constants import (DEFAULT_DOMAIN_PATH, DEFAULT_CONFIG_PATH,
                            DEFAULT_MODELS_PATH, DEFAULT_DATA_PATH)


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
    arguments.training_files = os.path.join(project, DEFAULT_DATA_PATH)
    arguments.force = False

    train(arguments)

    return arguments.out


@pytest.fixture(scope="session")
def trained_model(project) -> Text:
    return train_model(project)

