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
    import rasa.train

    output = os.path.join(project, DEFAULT_MODELS_PATH, filename)
    domain = os.path.join(project, DEFAULT_DOMAIN_PATH)
    config = os.path.join(project, DEFAULT_CONFIG_PATH)
    training_files = os.path.join(project, DEFAULT_DATA_PATH)

    rasa.train(domain, config, training_files, output)

    return output


@pytest.fixture(scope="session")
def trained_model(project) -> Text:
    return train_model(project)
