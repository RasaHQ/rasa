import copy
import re
import warnings
from typing import Any, Dict, Text

import pytest
from pytest import Testdir

import rasa.cli.data
import rasa.cli.scaffold
import rasa.cli.shell
import rasa.cli.train
import rasa.cli.utils
import rasa.shared.utils.io
from rasa.__main__ import create_argument_parser
from rasa.cli.validation.bot_config import validate_files
from rasa.core.config.configuration import (
    Configuration,
    CredentialsConfigPath,
    EndpointsConfigPath,
    MessageProcessingConfigPath,
)
from rasa.shared.constants import ASSISTANT_ID_KEY
from rasa.utils.common import EXPECTED_WARNINGS
from rasa.utils.io import write_yaml


@pytest.mark.timeout(300, func_only=True)
def test_default_project_has_no_warnings(
    testdir: Testdir, default_config: Dict[Text, Any]
):
    parser = create_argument_parser()
    rasa.cli.scaffold.create_initial_project(".")

    config = copy.deepcopy(default_config)
    # change default assistant id value to prevent config validation errors
    config[ASSISTANT_ID_KEY] = "some_unique_assistant_name"

    write_yaml(config, "config.yml")

    Configuration.initialise_endpoints(
        EndpointsConfigPath.default_file_path(),
    ).initialise_credentials(
        CredentialsConfigPath.default_file_path(),
    ).initialise_message_processing(
        MessageProcessingConfigPath.default_file_path(),
    )

    # Clear any existing warnings before starting the test
    warnings.resetwarnings()

    # Record warnings, but do not raise exception if no warnings are recorded.
    # Use a more specific warning filter to avoid catching warnings from other tests
    with warnings.catch_warnings(record=True) as warning_recorder:
        warnings.simplefilter("always")
        arg_namespace = parser.parse_args(["data", "validate"])
        validate_files(
            arg_namespace.fail_on_warnings,
            arg_namespace.max_history,
            rasa.cli.data._build_training_data_importer(arg_namespace),
        )
        rasa.cli.train.run_training(parser.parse_args(["train"]))

    # Filter out expected warnings and check for unexpected ones
    unexpected_warnings = [
        warning.message
        for warning in warning_recorder
        if not any(
            type(warning.message) == warning_type
            and re.search(warning_message, str(warning.message))
            for warning_type, warning_message in EXPECTED_WARNINGS
        )
    ]

    # Only fail if there are unexpected warnings
    assert not unexpected_warnings
