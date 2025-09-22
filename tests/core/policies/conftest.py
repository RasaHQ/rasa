import pytest

from rasa.core.config.configuration import Configuration


@pytest.fixture(scope="package", autouse=True)
def default_config() -> Configuration:
    return Configuration.initialise_empty()
