import os
import sys
from pathlib import Path
from typing import Callable

import pytest
from _pytest.pytester import RunResult

from rasa.shared.constants import ASSISTANT_ID_KEY
from tests.cli.conftest import RASA_EXE, create_simple_project_with_missing_assistant_id


def test_shell_help(run: Callable[..., RunResult]):
    output = run("shell", "--help")

    if sys.version_info.minor >= 9:
        # This is required because `argparse` behaves differently on
        # Python 3.9 and above. The difference is the changed formatting of help
        # output for CLI arguments with `nargs="*"
        version_dependent = """[-i INTERFACE] [-p PORT] [-t AUTH_TOKEN] [--cors [CORS ...]]
                  [--enable-api] [--response-timeout RESPONSE_TIMEOUT]"""  # noqa: E501
    else:
        version_dependent = """[-i INTERFACE] [-p PORT] [-t AUTH_TOKEN]
                  [--cors [CORS [CORS ...]]] [--enable-api]
                  [--response-timeout RESPONSE_TIMEOUT]"""

    help_text = (
        f"""usage: {RASA_EXE} shell [-h] [-v] [-vv] [--quiet]
                  [--logging-config-file LOGGING_CONFIG_FILE]
                  [--conversation-id CONVERSATION_ID] [-m MODEL]
                  [--log-file LOG_FILE] [--use-syslog]
                  [--syslog-address SYSLOG_ADDRESS]
                  [--syslog-port SYSLOG_PORT]
                  [--syslog-protocol SYSLOG_PROTOCOL] [--endpoints ENDPOINTS]
                  """
        + version_dependent
        + """
                  [--remote-storage REMOTE_STORAGE]
                  [--ssl-certificate SSL_CERTIFICATE]
                  [--ssl-keyfile SSL_KEYFILE] [--ssl-ca-file SSL_CA_FILE]
                  [--ssl-password SSL_PASSWORD] [--credentials CREDENTIALS]
                  [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
                  [--jwt-method JWT_METHOD]
                  {nlu} ... [model-as-positional-argument]"""
    )

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_shell_nlu_help(run: Callable[..., RunResult]):
    output = run("shell", "nlu", "--help")

    help_text = f"""usage: {RASA_EXE} shell nlu [-h] [-v] [-vv] [--quiet]
                      [--logging-config-file LOGGING_CONFIG_FILE] [-m MODEL]
                      [model-as-positional-argument]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


# FIXME: this test passes locally but fails in the CI with timeout > 300s
@pytest.mark.skip_on_ci
async def test_shell_without_assistant_id_issues_warning(
    tmp_path: Path, trained_async: Callable, run: Callable[..., RunResult]
):
    os.environ["LOG_LEVEL"] = "ERROR"

    create_simple_project_with_missing_assistant_id(tmp_path)

    domain_path = tmp_path / "domain.yml"
    stories_path = tmp_path / "data" / "stories.yml"
    nlu_path = tmp_path / "data" / "nlu.yml"
    config_path = tmp_path / "config.yml"

    warning_message = (
        f"The model metadata does not contain a value for the '{ASSISTANT_ID_KEY}' "
        f"attribute. Check that 'config.yml' file contains a value for "
        f"the '{ASSISTANT_ID_KEY}' key and re-train the model."
    )

    model_path = await trained_async(
        domain=domain_path, config=config_path, training_files=[stories_path, nlu_path]
    )

    output = run("shell", "--model", str(model_path))

    printed_output = {line.strip() for line in output.outlines}

    assert any([warning_message in line for line in printed_output])
