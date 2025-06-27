from pathlib import Path
from typing import Dict, Optional, Text

import pytest

import rasa.studio.prompts as prompts
from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.dialogue_understanding.generator import SingleStepLLMCommandGenerator
from rasa.shared.utils.yaml import read_yaml, write_yaml
from rasa.studio.data_handler import StudioDataHandler


class _DummyHandler(StudioDataHandler):
    def __init__(self, returned_prompts: Optional[Dict[Text, Text]]):
        self._prompts = returned_prompts

    def get_prompts(self) -> Optional[Dict[str, str]]:
        return self._prompts


@pytest.fixture()
def empty_project(tmp_path: Path) -> Path:
    """Create a minimal project with empty config / endpoints files."""
    (tmp_path / prompts.DEFAULT_CONFIG_PATH).touch()
    (tmp_path / prompts.DEFAULT_ENDPOINTS_PATH).touch()
    return tmp_path


def test_handle_prompts_no_prompts(empty_project: Path):
    handler = _DummyHandler(returned_prompts=None)
    prompts.handle_prompts(handler.get_prompts(), empty_project)

    # config / endpoints remain empty
    assert not (empty_project / prompts.DEFAULT_CONFIG_PATH).read_text()
    assert not (empty_project / prompts.DEFAULT_ENDPOINTS_PATH).read_text()

    # no prompts directory created
    assert not (empty_project / prompts.DEFAULT_PROMPTS_PATH).exists()


def test_handle_prompts_all_custom(empty_project: Path, monkeypatch):
    # Studio data handler should return custom prompts
    prompt_names = [
        prompts.CONTEXTUAL_RESPONSE_REPHRASER_NAME,
        prompts.COMMAND_GENERATOR_NAME,
        prompts.ENTERPRISE_SEARCH_NAME,
    ]
    custom = {prompt_name: prompt_name for prompt_name in prompt_names}
    handler = _DummyHandler(returned_prompts=custom)

    # Create a config.yml file with a command generator and an enterprise search policy
    config = {
        prompts.CONFIG_PIPELINE_KEY: [{"name": SingleStepLLMCommandGenerator.__name__}],
        prompts.CONFIG_POLICIES_KEY: [{"name": EnterpriseSearchPolicy.__name__}],
    }
    write_yaml(data=config, target=empty_project / prompts.DEFAULT_CONFIG_PATH)

    prompts.handle_prompts(handler.get_prompts(), empty_project)

    # Prompts directory should be created for custom prompts
    prompt_dir = empty_project / prompts.DEFAULT_PROMPTS_PATH
    assert prompt_dir.exists()

    # Confirm all custom prompts are written to files
    expected_files = {
        f"{prompt_name}.jinja2": prompt_name for prompt_name in prompt_names
    }
    for prompt_file_name, prompt_content in expected_files.items():
        prompt_path = prompt_dir / prompt_file_name
        assert prompt_path.is_file()
        assert prompt_path.read_text(encoding="utf-8") == prompt_content

    # Confirm that endpoints.yml file includes custom prompt path
    endpoints = read_yaml(empty_project / prompts.DEFAULT_ENDPOINTS_PATH)
    rephraser_prompt_path = str(
        Path(prompts.DEFAULT_PROMPTS_PATH)
        / f"{prompts.CONTEXTUAL_RESPONSE_REPHRASER_NAME}.jinja2"
    )
    assert endpoints["nlg"]["prompt"] == rephraser_prompt_path

    # Confirm that config.yml file includes custom prompt paths
    config = read_yaml(empty_project / prompts.DEFAULT_CONFIG_PATH)
    command_generator_prompt_path = str(
        Path(prompts.DEFAULT_PROMPTS_PATH) / f"{prompts.COMMAND_GENERATOR_NAME}.jinja2"
    )
    assert (
        config[prompts.CONFIG_PIPELINE_KEY][0][prompts.PROMPT_TEMPLATE_CONFIG_KEY]
        == command_generator_prompt_path
    )

    enterprise_search_prompt_path = str(
        Path(prompts.DEFAULT_PROMPTS_PATH) / f"{prompts.ENTERPRISE_SEARCH_NAME}.jinja2"
    )
    assert (
        config[prompts.CONFIG_POLICIES_KEY][0][prompts.PROMPT_CONFIG_KEY]
        == enterprise_search_prompt_path
    )


@pytest.mark.parametrize(
    "studio,system,expected",
    [
        ("foo", "bar", True),
        ("foo", "foo", False),
        (None, "foo", False),
    ],
)
def test_is_custom_prompt(studio, system, expected):
    assert prompts._is_custom_prompt(studio, system) is expected
