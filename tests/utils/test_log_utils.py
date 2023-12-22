import os
import logging
from unittest import mock
import pytest

from rasa.utils.log_utils import configure_structlog, log_llm

import structlog
from structlog import get_logger
from structlog.testing import capture_logs

# from structlog.testing import LogCapture

# @pytest.fixture(name="log_output")
# def fixture_log_output():
#     return LogCapture()


logging.basicConfig(level="INFO")

log_event = "some log event text"
attribute = "some attribute text"


@pytest.mark.parametrize(
    ("environment_variables, logging_output"),
    [
        ({}, []),
        (
            {"LOG_LEVEL_LLM": "INFO"},
            [{"event": log_event, "log_level": "info", "attribute": attribute}],
        ),
        (
            {
                "LOG_LEVEL_LLM": "DEBUG",
                "LOG_LEVEL_LLM_COMMAND_GENERATOR": "INFO",
            },
            [{"event": log_event, "log_level": "info", "attribute": attribute}],
        ),
    ],
    ids=[
        "LLM logging environment variables not specified",
        "Only LOG_LEVEL_LLM='DEBUG'",
        "Only LOG_LEVEL_LLM_COMMAND_GENERATOR='DEBUG'",
    ],
)
def test_log_llm(environment_variables, logging_output):
    """Environment variables control the llm logging as expected"""
    configure_structlog(log_level=20)  # DEBUG=10, INFO=20

    with capture_logs() as cap_logs:
        with mock.patch.dict(
            os.environ,
            environment_variables,
        ):
            logger = get_logger()
            log_llm(
                logger=logger,
                log_module="LLMCommandGenerator",
                log_event=log_event,
                attribute=attribute,
            )
            assert cap_logs == logging_output
