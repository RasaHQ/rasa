from typing import Any, Dict, Text

from _pytest.pytester import Testdir
import pytest
import copy

import re

from rasa.__main__ import create_argument_parser
import rasa.cli.data
import rasa.cli.scaffold
import rasa.cli.train
import rasa.cli.shell
import rasa.shared.utils.io
from rasa.utils.common import EXPECTED_WARNINGS


@pytest.mark.timeout(300, func_only=True)
def test_default_project_has_no_warnings(
    testdir: Testdir, default_config: Dict[Text, Any]
):
    parser = create_argument_parser()
    rasa.cli.scaffold.create_initial_project(".")

    config = copy.deepcopy(default_config)
    for _, items in config.items():
        for item in items:
            if "epochs" in item:
                item["epochs"] = 1
                item["evaluate_every_number_of_epochs"] = -1

    rasa.shared.utils.io.write_yaml(config, "config.yml")

    with pytest.warns(None) as warning_recorder:
        rasa.cli.data.validate_files(parser.parse_args(["data", "validate"]))
        rasa.cli.train.run_training(parser.parse_args(["train"]))

    # pytest.warns would override any warning filters that we could set
    assert not [
        warning
        for warning in warning_recorder.list
        if not any(
            type(warning.message) == warning_type
            and re.match(warning_message, str(warning.message))
            for warning_type, warning_message in EXPECTED_WARNINGS
        )
    ]
