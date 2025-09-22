from typing import Callable, Dict, Optional
from unittest.mock import MagicMock, Mock

import pytest
from pytest import RunResult

from rasa.cli.dialogue_understanding_test import _get_llm_command_generator_config
from rasa.dialogue_understanding.generator import (
    MultiStepLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.constants import DEFAULT_LLM_CONFIG
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.shared.utils.llm import combine_custom_and_default_config
from tests.conftest import get_model_groups


def test_rasa_test_dialogue_understanding_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa test du [-h] [-v] [-vv] [--quiet]
                    [--logging-config-file LOGGING_CONFIG_FILE]
                    [--output-file OUTPUT_FILE] [--no-output]
                    [-m MODEL] [--endpoints ENDPOINTS]
                    [--output-prompt]
                    [--remote-storage REMOTE_STORAGE]
                    [path-to-test-cases]
                    [--remove-default-commands [REMOVE_DEFAULT_COMMANDS ...]]
                    [--additional-commands [ADDITIONAL_COMMANDS ...]]

Runs dialogue understanding testing."""
    lines = help_text.split("\n")

    output = run("test", "du", "--help")

    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


@pytest.mark.parametrize(
    "llm_command_generator_node, expected_llm_config",
    [
        # Graph schema with SingleStepLLMCommandGenerator with deprecated LLM config
        (
            SchemaNode(
                needs={},
                uses=SingleStepLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"provider": "openai", "model": "test-gpt"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                {"provider": "openai", "model": "test-gpt"}, DEFAULT_LLM_CONFIG
            ),
        ),
        # Graph schema with SingleStepLLMCommandGenerator with model groups LLM config
        (
            SchemaNode(
                needs={},
                uses=SingleStepLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"model_group": "llm-model-group"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                get_model_groups()[0],
                DEFAULT_LLM_CONFIG,
            ),
        ),
        # Graph schema with MultiStepLLMCommandGenerator with deprecated LLM config
        (
            SchemaNode(
                needs={},
                uses=MultiStepLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"provider": "openai", "model": "test-gpt"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                {"provider": "openai", "model": "test-gpt"}, DEFAULT_LLM_CONFIG
            ),
        ),
        # Graph schema with MultiStepLLMCommandGenerator with model groups LLM config
        (
            SchemaNode(
                needs={},
                uses=MultiStepLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"model_group": "llm-model-group"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                get_model_groups()[0],
                DEFAULT_LLM_CONFIG,
            ),
        ),
        # Graph schema without any LLMCommandGenerator
        (
            None,
            None,
        ),
    ],
)
def test_get_llm_command_generator_config(
    llm_command_generator_node: Optional[SchemaNode],
    expected_llm_config: Optional[Dict],
    mock_configuration: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
):
    # Given
    graph_schema_nodes = {
        "test_node_1": SchemaNode(
            needs={}, uses=Mock, constructor_name="create", fn="train", config={}
        ),
        "test_node_2": SchemaNode(
            needs={}, uses=Mock, constructor_name="create", fn="train", config={}
        ),
    }
    if llm_command_generator_node is not None:
        graph_schema_nodes["test_SingleStepLLMCommandGenerator_3"] = (
            llm_command_generator_node
        )

    test_runner = Mock()
    test_runner.agent.processor.model_metadata.train_schema = GraphSchema(
        graph_schema_nodes
    )
    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    # When
    result = _get_llm_command_generator_config(test_runner.agent.processor)
    # Then
    assert result == expected_llm_config
