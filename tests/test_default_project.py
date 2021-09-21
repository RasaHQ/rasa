import copy
from typing import Any, Dict, Text

from _pytest.pytester import Testdir
import pytest

from rasa.__main__ import create_argument_parser
from rasa.cli.arguments.data import set_validator_arguments
import rasa.cli.data
import rasa.cli.scaffold
import rasa.cli.train
from rasa.shared.importers.rasa import RasaFileImporter
import rasa.shared.utils.io


@pytest.mark.timeout(300, func_only=True)
def test_default_project_has_no_warnings(
    testdir: Testdir, default_config: Dict[Text, Any]
):
    parser = create_argument_parser()
    rasa.cli.scaffold.create_initial_project('.')

    # TODO: put somewhere shareable?
    config = copy.deepcopy(default_config)
    for model_part, items in config.items():
        for item in items:
            if "epochs" in item:
                item["epochs"] = 1
                item["evaluate_every_number_of_epochs"] = -1

    rasa.shared.utils.io.write_yaml(config, "config.yml")

    with pytest.warns(None) as warnings:
        rasa.cli.data.validate_files(parser.parse_args(['data', 'validate']))
        rasa.cli.train.run_training(parser.parse_args(['train']))

    print([t.__str__() for t in warnings._list])

    assert not warnings
