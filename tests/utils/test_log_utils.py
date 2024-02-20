import os
import logging
from unittest import mock
import pytest

from rasa.utils.log_utils import log_llm

from structlog import get_logger
from structlog.testing import capture_logs

logging.basicConfig(level="INFO")

log_event = "some log event text"
attribute = "some attribute text"


@pytest.mark.parametrize(
    ("log_module, environment_variables, logging_output"),
    [
        (
            "LLMCommandGenerator",
            {},
            [{"event": log_event, "log_level": "debug", "attribute": attribute}],
        ),
        (
            "LLMCommandGenerator",
            {
                "LOG_LEVEL_LLM": "INFO",
                "LOG_LEVEL_LLM_COMMAND_GENERATOR": "DEBUG",
            },
            [{"event": log_event, "log_level": "info", "attribute": attribute}],
        ),
        (
            "LLMCommandGenerator",
            {
                "LOG_LEVEL_LLM": "DEBUG",
                "LOG_LEVEL_LLM_COMMAND_GENERATOR": "INFO",
            },
            [{"event": log_event, "log_level": "info", "attribute": attribute}],
        ),
        (
            "CustomLLMCommandGenerator",
            {},
            [{"event": log_event, "log_level": "debug", "attribute": attribute}],
        ),
        (
            "CustomLLMCommandGenerator",
            {"LOG_LEVEL_LLM_CUSTOMLLMCOMMANDGENERATOR": "INFO"},
            [{"event": log_event, "log_level": "info", "attribute": attribute}],
        ),
    ],
    ids=[
        "LLM logging environment variables not specified",
        "Only LOG_LEVEL_LLM='INFO'",
        "Only LOG_LEVEL_LLM_COMMAND_GENERATOR='INFO'",
        "Custom Component",
        "Custom Component with INFO Level logging",
    ],
)
def test_log_llm(log_module, environment_variables, logging_output):
    """Check that environment variables control the llm logging as expected."""

    with capture_logs() as cap_logs:
        with mock.patch.dict(
            os.environ,
            environment_variables,
        ):
            logger = get_logger()
            log_llm(
                logger=logger,
                log_module=log_module,
                log_event=log_event,
                attribute=attribute,
            )
            assert cap_logs == logging_output
