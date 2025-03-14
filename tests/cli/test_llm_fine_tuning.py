import argparse
from typing import Any, Callable, Dict, List, Optional, Text, Union
from unittest.mock import Mock, patch

import pytest
from pytest import RunResult

from rasa.cli.llm_fine_tuning import (
    PARAMETERS_FILE,
    RESULT_SUMMARY_FILE,
    _get_llm_command_generator_config,
    create_storage_context,
    restricted_float,
    write_params,
    write_statistics,
)
from rasa.dialogue_understanding.generator import (
    CompactLLMCommandGenerator,
    MultiStepLLMCommandGenerator,
    SingleStepLLMCommandGenerator,
)
from rasa.dialogue_understanding.generator.constants import DEFAULT_LLM_CONFIG
from rasa.engine.graph import GraphSchema, SchemaNode
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.llm_fine_tuning.storage import (
    FileStorageStrategy,
    StorageContext,
    StorageType,
)
from rasa.shared.utils.llm import combine_custom_and_default_config


class MockSingleStepLLMCommandGenerator(SingleStepLLMCommandGenerator):
    """A mock of what would a custom SSLLMCG that inherits
    from the original look like.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, model_storage, resource, prompt_template)

    async def invoke_llm(
        self, prompt: Union[List[dict], List[str], str]
    ) -> Optional[str]:
        pass


class MockAvailableEndpoints:
    @staticmethod
    def get_instance():
        return MockAvailableEndpoints()

    def __init__(self):
        self.model_groups = [
            {
                "id": "llm-model-group",
                "models": [
                    {
                        "provider": "cohere",
                        "model": "test-cohere",
                        "api_key": "mock key in test_tracing_rephraser",
                    },
                    {
                        "provider": "openai",
                        "model": "gpt-4",
                        "api_key": "tedst",
                    },
                    {
                        "provider": "azure",
                        "deployment": "my-llm-azure-deployment",
                        "api_key": "test",
                        "api_base": "test-base",
                        "api_version": "test-version",
                        "num_retries": 100,
                        "timeout": 100,
                    },
                ],
                "router": {"routing_strategy": "test"},
            },
        ]


def test_rasa_llm(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa llm [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE] {finetune}
    """
    lines = help_text.split("\n")

    output = run("llm", "--help")

    printed_help = [line.strip() for line in output.outlines]
    printed_help = str.join(" ", printed_help)  # type: ignore
    for line in lines:
        assert line.strip() in printed_help


def test_rasa_finetune_llm(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa llm finetune [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE] {prepare-data}
    """
    lines = help_text.split("\n")

    output = run("llm", "finetune", "--help")

    printed_help = [line.strip() for line in output.outlines]
    printed_help = str.join(" ", printed_help)  # type: ignore
    for line in lines:
        assert line.strip() in printed_help


def test_rasa_finetune_llm_prepare_data(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa llm finetune prepare-data [-h] [-v] [-vv] [--quiet]
    [--logging-config-file LOGGING_CONFIG_FILE] [-o OUT]
    [--remote-storage REMOTE_STORAGE]
    [--num-rephrases {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
    25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49}]
    [--rephrase-config REPHRASE_CONFIG]
    [--train-frac TRAIN_FRAC]
    [--output-format [{instruction,conversational}]]
    [-m MODEL]
    [--endpoints ENDPOINTS]
    [path-to-e2e-test-cases]
    """
    lines = help_text.split("\n")

    output = run("llm", "finetune", "prepare-data", "--help")

    printed_help = [line.strip() for line in output.outlines]
    printed_help = str.join(" ", printed_help)  # type: ignore
    for line in lines:
        assert line.strip() in printed_help


@pytest.fixture
def args():
    mock_args = argparse.Namespace()
    mock_args.out = "output_test"
    mock_args.num_rephrases = 10
    mock_args.rephrase_config = "rephrasing_config.yaml"
    mock_args.train_frac = 0.8
    mock_args.output_format = "alpaca"
    mock_args.model = "dummy_model"
    mock_args.endpoints = "dummy_endpoints"
    mock_args.remote_storage = None
    mock_args.path_to_e2e_test_cases = "e2e_tests"
    return mock_args


def test_restricted_float():
    assert restricted_float(0.5) == 0.5
    assert restricted_float(1.0) == 1.0

    with pytest.raises(argparse.ArgumentTypeError):
        restricted_float(0.0)

    with pytest.raises(argparse.ArgumentTypeError):
        restricted_float(1.1)

    with pytest.raises(argparse.ArgumentTypeError):
        restricted_float("invalid")


@patch("rasa.shared.utils.yaml.write_yaml")
def test_write_params(mock_write_yaml, args):
    rephrase_config = {"some_key": "some_value"}

    write_params(args, rephrase_config, args.out)
    yaml_data = {
        "parameters": {
            "num_rephrases": args.num_rephrases,
            "rephrase_config": rephrase_config,
            "model": args.model,
            "endpoints": args.endpoints,
            "remote-storage": args.remote_storage,
            "train_frac": args.train_frac,
            "output_format": args.output_format,
            "out": args.out,
        }
    }
    mock_write_yaml.assert_called_once_with(yaml_data, f"{args.out}/{PARAMETERS_FILE}")


@patch("rasa.shared.utils.yaml.write_yaml")
def test_write_statistics(mock_write_yaml, args):
    statistics = {"stat1": 1, "stat2": 2}

    write_statistics(statistics, args.out)
    mock_write_yaml.assert_called_once_with(
        statistics, f"{args.out}/{RESULT_SUMMARY_FILE}"
    )


def test_create_storage_context():
    context = create_storage_context(StorageType.FILE, "output")

    assert isinstance(context, StorageContext) is True
    assert isinstance(context.strategy, FileStorageStrategy) is True
    assert context.strategy.output_dir == "output"


@pytest.mark.parametrize(
    "single_step_llm_command_generator_node,"
    "expected_llm_config,"
    "should_raise_an_error",
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
            False,
        ),
        # Graph schema with the custom SingleStepLLMCommandGenerator with deprecated LLM
        # config
        (
            SchemaNode(
                needs={},
                uses=MockSingleStepLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"provider": "openai", "model": "test-gpt"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                {"provider": "openai", "model": "test-gpt"}, DEFAULT_LLM_CONFIG
            ),
            False,
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
                MockAvailableEndpoints.get_instance().model_groups[0],
                DEFAULT_LLM_CONFIG,
            ),
            False,
        ),
        # Graph schema with a custom SingleStepLLMCommandGenerator with model groups LLM
        # config
        (
            SchemaNode(
                needs={},
                uses=MockSingleStepLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"model_group": "llm-model-group"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                MockAvailableEndpoints.get_instance().model_groups[0],
                DEFAULT_LLM_CONFIG,
            ),
            False,
        ),
        # Graph schema without SingleStepLLMCommandGenerator
        (
            None,
            None,
            True,
        ),
        # Graph schema with MultiStepLLMCommandGenerator should fail, as it's not
        # supported for now
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
            None,
            True,
        ),
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
            None,
            True,
        ),
    ],
)
def test_get_llm_command_generator_config(
    single_step_llm_command_generator_node: SchemaNode,
    expected_llm_config: dict,
    should_raise_an_error: bool,
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
    if single_step_llm_command_generator_node is not None:
        graph_schema_nodes["test_SingleStepLLMCommandGenerator_3"] = (
            single_step_llm_command_generator_node
        )

    e2e_test_runner = Mock()
    e2e_test_runner.agent.processor.model_metadata.train_schema = GraphSchema(
        graph_schema_nodes
    )
    mock_endpoints = MockAvailableEndpoints()
    monkeypatch.setattr("rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints)

    if not should_raise_an_error:
        # When
        result = _get_llm_command_generator_config(e2e_test_runner)
        # Then A
        assert result == expected_llm_config
    else:
        # Then B
        with pytest.raises(SystemExit):
            _get_llm_command_generator_config(e2e_test_runner)


@pytest.mark.parametrize(
    "compact_llm_command_generator_node,"
    "expected_llm_config,"
    "should_raise_an_error",
    [
        # Graph schema with CompactLLMCommandGenerator with deprecated LLM config
        (
            SchemaNode(
                needs={},
                uses=CompactLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"provider": "openai", "model": "test-gpt"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                {"provider": "openai", "model": "test-gpt"}, DEFAULT_LLM_CONFIG
            ),
            False,
        ),
        # Graph schema with CompactLLMCommandGenerator with model groups LLM config
        (
            SchemaNode(
                needs={},
                uses=CompactLLMCommandGenerator,
                constructor_name="create",
                fn="train",
                config={"llm": {"model_group": "llm-model-group"}},
                is_target=True,
                is_input=False,
            ),
            combine_custom_and_default_config(
                MockAvailableEndpoints.get_instance().model_groups[0],
                DEFAULT_LLM_CONFIG,
            ),
            False,
        ),
        # Graph schema without CompactLLMCommandGenerator
        (
            None,
            None,
            True,
        ),
    ],
)
def test_get_llm_command_generator_config_for_compact_llm_command_generator(
    compact_llm_command_generator_node: SchemaNode,
    expected_llm_config: dict,
    should_raise_an_error: bool,
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
    if compact_llm_command_generator_node is not None:
        graph_schema_nodes["test_CompactLLMCommandGenerator_3"] = (
            compact_llm_command_generator_node
        )

    e2e_test_runner = Mock()
    e2e_test_runner.agent.processor.model_metadata.train_schema = GraphSchema(
        graph_schema_nodes
    )
    mock_endpoints = MockAvailableEndpoints()
    monkeypatch.setattr("rasa.shared.utils.llm.AvailableEndpoints", mock_endpoints)

    if not should_raise_an_error:
        # When
        result = _get_llm_command_generator_config(e2e_test_runner)
        # Then A
        assert result == expected_llm_config
    else:
        # Then B
        with pytest.raises(SystemExit):
            _get_llm_command_generator_config(e2e_test_runner)
