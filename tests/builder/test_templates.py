from pathlib import Path

import pytest

from rasa.cli.scaffold import ProjectTemplateName
from rasa.shared.constants import ASSISTANT_ID_DEFAULT_VALUE
from rasa.shared.utils.yaml import read_yaml


@pytest.mark.parametrize("template", ProjectTemplateName.get_all_values())
def test_template_use_the_default_id_in_config(template: ProjectTemplateName) -> None:
    # get the project folder
    project_folder = Path(f"rasa/cli/project_templates/{template}")
    # get the config file
    config_file = project_folder / "config.yml"
    # read the config file
    config = read_yaml(config_file)
    # check that the assistant_id is the default value
    assert config["assistant_id"] == ASSISTANT_ID_DEFAULT_VALUE
