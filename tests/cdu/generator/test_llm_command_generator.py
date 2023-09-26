import uuid
from typing import Dict

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


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {"prompt": "data/test_prompt_templates/test_prompt.jinja2"},
            "This is a test prompt.",
        ),
        (
            {},
            "Your task is to analyze the current conversation",
        ),
    ],
)
async def test_llm_command_generator_prompt_initialisation(
    model_storage: ModelStorage, resource: Resource, config: Dict, expected: str
):
    generator = LLMCommandGenerator(config, model_storage, resource)
    assert generator.prompt_template.startswith(expected)
