import uuid

import pytest
from _pytest.tmpdir import TempPathFactory

from rasa.dialogue_understanding.generator.llm_command_generator import (
    LLMCommandGenerator,
)
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage


@pytest.fixture(scope="session")
def resource() -> Resource:
    return Resource(uuid.uuid4().hex)


@pytest.fixture(scope="session")
def model_storage(tmp_path_factory: TempPathFactory) -> ModelStorage:
    return LocalModelStorage(tmp_path_factory.mktemp(uuid.uuid4().hex))


async def test_llm_command_generator_prompt_init_custom(
    model_storage: ModelStorage, resource: Resource
) -> None:
    generator = LLMCommandGenerator(
        {"prompt": "data/test_prompt_templates/test_prompt.jinja2"},
        model_storage,
        resource,
    )
    assert generator.prompt_template.startswith("This is a test prompt.")


async def test_llm_command_generator_prompt_init_default(
    model_storage: ModelStorage, resource: Resource
) -> None:
    generator = LLMCommandGenerator({}, model_storage, resource)
    assert generator.prompt_template.startswith(
        "Your task is to analyze the current conversation"
    )
