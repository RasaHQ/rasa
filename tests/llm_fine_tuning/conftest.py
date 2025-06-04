import textwrap
import uuid
from typing import Text

import pytest
from pytest import TempPathFactory

import rasa.shared.utils.io
from rasa.core.agent import Agent
from rasa.core.available_endpoints import AvailableEndpoints
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from tests.conftest import TrainedAsync


@pytest.fixture(scope="package")
def domain_path_for_finetuning() -> Text:
    return "data/test_llm_finetuning/domain.yml"


@pytest.fixture(scope="package")
def flows_path_for_finetuning() -> Text:
    return "data/test_llm_finetuning/flows.yml"


@pytest.fixture(scope="package")
def trained_async_llm(
    trained_async: TrainedAsync,
    llm_endpoints: AvailableEndpoints,  # or llm_endpoints if that creates it
) -> TrainedAsync:
    return trained_async  # or return a wrapper around it


@pytest.fixture(scope="package")
async def trained_single_step_agent_model(
    trained_async_llm: TrainedAsync,
    domain_path_for_finetuning: Text,
    flows_path_for_finetuning: Text,
    single_step_config_path: Text,
) -> Text:
    model_path = await trained_async_llm(
        domain_path_for_finetuning,
        single_step_config_path,
        training_files=[flows_path_for_finetuning],
    )

    return model_path


@pytest.fixture(scope="package")
async def trained_compact_agent_model(
    trained_async_llm: TrainedAsync,
    domain_path_for_finetuning: Text,
    flows_path_for_finetuning: Text,
    compact_config_path: Text,
) -> Text:
    model_path = await trained_async_llm(
        domain_path_for_finetuning,
        compact_config_path,
        training_files=[flows_path_for_finetuning],
    )

    return model_path


@pytest.fixture(scope="package")
def single_step_config_path(tmp_path_factory: TempPathFactory) -> Text:
    project_path = tmp_path_factory.mktemp(uuid.uuid4().hex)

    config = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        recipe: default.v1
        language: en
        pipeline:
        - name: SingleStepLLMCommandGenerator
          llm:
            model_group: rasa_command_generation_model
          flow_retrieval:
            active: false

        policies:
        - name: FlowPolicy
        """
    )
    config_path = project_path / "config.yml"
    rasa.shared.utils.io.write_text_file(config, config_path)

    return str(config_path)


@pytest.fixture(scope="package")
def compact_config_path(tmp_path_factory: TempPathFactory) -> Text:
    project_path = tmp_path_factory.mktemp(uuid.uuid4().hex)

    config = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        recipe: default.v1
        language: en
        pipeline:
        - name: CompactLLMCommandGenerator
          llm:
            model_group: rasa_command_generation_model
          flow_retrieval:
            active: false

        policies:
        - name: FlowPolicy
        """
    )
    config_path = project_path / "config.yml"
    rasa.shared.utils.io.write_text_file(config, config_path)

    return str(config_path)


@pytest.fixture(scope="package")
def llm_endpoints(tmp_path_factory: TempPathFactory) -> AvailableEndpoints:
    project_path = tmp_path_factory.mktemp(uuid.uuid4().hex)

    endpoints = textwrap.dedent(
        """
        action_endpoint:
          url: "http://localhost:5055/webhook"

        nlg:
          type: rephrase

        model_groups:
          - id: rasa_command_generation_model
            models:
              - provider: rasa
                model: rasa/cmd_gen_codellama_13b_calm_demo
                api_base: "https://tutorial-llm.rasa.ai"
        """
    )
    endpoints_path = project_path / "endpoints.yml"
    rasa.shared.utils.io.write_text_file(endpoints, endpoints_path)

    AvailableEndpoints.reset_instance()
    llm_endpoints = AvailableEndpoints.get_instance(str(endpoints_path))

    return llm_endpoints


@pytest.fixture(scope="package")
def single_step_agent(
    llm_endpoints: AvailableEndpoints, trained_single_step_agent_model: Text
) -> Agent:
    return Agent.load(trained_single_step_agent_model, endpoints=llm_endpoints)


@pytest.fixture(scope="package")
def compact_agent(
    llm_endpoints: AvailableEndpoints, trained_compact_agent_model: Text
) -> Agent:
    return Agent.load(trained_compact_agent_model, endpoints=llm_endpoints)
