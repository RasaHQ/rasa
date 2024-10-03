from pathlib import Path

import pytest
from _pytest.tmpdir import TempPathFactory

from rasa.core import utils
from rasa.shared.utils.io import write_text_file


@pytest.fixture(scope="session")
def empty_config_file(tmp_path_factory: TempPathFactory) -> Path:
    config_path = tmp_path_factory.getbasetemp() / "config.yml"
    utils.dump_obj_as_yaml_to_file(config_path, {})
    return config_path


@pytest.fixture(scope="session")
def small_domain_file(tmp_path_factory: TempPathFactory) -> Path:
    domain_content = """
        version: "2.0"
        responses:
            utter_greet:
            - text: hey there!
            - text: hey ho!
    """
    domain_path = tmp_path_factory.getbasetemp() / "domain.yml"
    write_text_file(domain_content, domain_path)
    return domain_path
