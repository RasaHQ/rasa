from pathlib import Path
from typing import Dict, List, Optional, Text

import structlog

from rasa.core.policies.enterprise_search_policy import EnterpriseSearchPolicy
from rasa.dialogue_understanding.generator.llm_based_command_generator import (
    LLMBasedCommandGenerator,
)
from rasa.shared.constants import (
    CONFIG_NAME_KEY,
    CONFIG_PIPELINE_KEY,
    CONFIG_POLICIES_KEY,
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_PROMPTS_PATH,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
)
from rasa.shared.utils.common import all_subclasses
from rasa.shared.utils.llm import get_system_default_prompts
from rasa.shared.utils.yaml import read_yaml, write_yaml

structlogger = structlog.get_logger()

CONTEXTUAL_RESPONSE_REPHRASER_NAME = "contextual_response_rephraser"
COMMAND_GENERATOR_NAME = "command_generator"
ENTERPRISE_SEARCH_NAME = "enterprise_search"


def handle_prompts(prompts: Dict[Text, Text], root: Path) -> None:
    """Handle prompts for the assistant.

    Args:
        prompts: A dict containing prompt names as keys and their content as values.
        root: The root directory where the prompts should be saved.
    """
    if not prompts:
        return

    config_path = root / DEFAULT_CONFIG_PATH
    endpoints_path = root / DEFAULT_ENDPOINTS_PATH
    config: Dict = read_yaml(config_path)
    endpoints: Dict = read_yaml(endpoints_path)

    # System default prompts are dependent on the endpoints
    system_prompts = get_system_default_prompts(config, endpoints)

    _handle_contextual_response_rephraser(
        root,
        prompts.get(CONTEXTUAL_RESPONSE_REPHRASER_NAME),
        system_prompts.contextual_response_rephraser,
        endpoints,
    )
    _handle_command_generator(
        root,
        prompts.get(COMMAND_GENERATOR_NAME),
        system_prompts.command_generator,
        config,
    )
    _handle_enterprise_search(
        root,
        prompts.get(ENTERPRISE_SEARCH_NAME),
        system_prompts.enterprise_search,
        config,
    )


def _handle_contextual_response_rephraser(
    root: Path,
    prompt_content: Optional[Text],
    system_prompt: Optional[Text],
    endpoints: Dict,
) -> None:
    """Handles the contextual response rephraser prompt.

    Args:
        root: The root directory where the prompt file will be saved.
        prompt_content: The content of the contextual response rephraser prompt.
        system_prompt: The system prompt for comparison.
        endpoints: The endpoints configuration to update with the prompt path.
    """
    if not _is_custom_prompt(prompt_content, system_prompt):
        return

    prompt_path = _save_prompt_file(
        root, f"{CONTEXTUAL_RESPONSE_REPHRASER_NAME}.jinja2", prompt_content
    )

    endpoints["nlg"] = endpoints.get("nlg") or {}
    endpoints["nlg"]["prompt"] = str(prompt_path)

    endpoints_path = root / DEFAULT_ENDPOINTS_PATH
    write_yaml(data=endpoints, target=endpoints_path, should_preserve_key_order=True)


def _handle_command_generator(
    root: Path,
    prompt_content: Optional[Text],
    system_prompt: Optional[Text],
    config: Dict,
) -> None:
    """Handles the command generator prompt.

    Args:
        root: The root directory where the prompt file will be saved.
        prompt_content: The content of the command generator prompt.
        system_prompt: The system prompt for comparison.
        config: The configuration dictionary to update with the prompt path.
    """
    if not _is_custom_prompt(prompt_content, system_prompt):
        return

    prompt_path = _save_prompt_file(
        root, f"{COMMAND_GENERATOR_NAME}.jinja2", prompt_content
    )

    command_generator_names: List[str] = [
        cls.__name__ for cls in all_subclasses(LLMBasedCommandGenerator)
    ]
    _add_prompt_to_config(
        config=config,
        section_key=CONFIG_PIPELINE_KEY,
        component_names=command_generator_names,
        prompt_key=PROMPT_TEMPLATE_CONFIG_KEY,
        prompt_path=str(prompt_path),
    )

    config_path = root / DEFAULT_CONFIG_PATH
    write_yaml(data=config, target=config_path, should_preserve_key_order=True)


def _handle_enterprise_search(
    root: Path,
    prompt_content: Optional[Text],
    system_prompt: Optional[Text],
    config: Dict,
) -> None:
    """Handles the enterprise search prompt.

    Args:
        root: The root directory where the prompt file will be saved.
        prompt_content: The content of the enterprise search prompt.
        system_prompt: The system prompt for comparison.
        config: The configuration dictionary to update with the prompt path.
    """
    if not _is_custom_prompt(prompt_content, system_prompt):
        return

    prompt_path = _save_prompt_file(
        root, f"{ENTERPRISE_SEARCH_NAME}.jinja2", prompt_content
    )

    _add_prompt_to_config(
        config=config,
        section_key=CONFIG_POLICIES_KEY,
        component_names=[EnterpriseSearchPolicy.__name__],
        prompt_key=PROMPT_CONFIG_KEY,
        prompt_path=str(prompt_path),
    )

    config_path = root / DEFAULT_CONFIG_PATH
    write_yaml(data=config, target=config_path, should_preserve_key_order=True)


def _is_custom_prompt(
    studio_prompt: Optional[Text], system_prompt: Optional[Text]
) -> bool:
    """Check if the prompt has been customized in Studio.

    Args:
        studio_prompt: The prompt content from the Studio.
        system_prompt: The default system prompt content.
    """
    return bool(studio_prompt and studio_prompt != system_prompt)


def _save_prompt_file(root: Path, filename: str, content: str) -> Path:
    """Save a prompt file to the specified root directory.

    Args:
        root: The root directory where the prompt file will be saved.
        filename: The name of the prompt file.
        content: The content of the prompt.
    """
    prompts_dir = root / DEFAULT_PROMPTS_PATH
    prompts_dir.mkdir(parents=True, exist_ok=True)

    file_path = prompts_dir / filename
    file_path.write_text(content, encoding="utf-8")

    return file_path.relative_to(root)


def _add_prompt_to_config(
    *,
    config: Dict,
    section_key: str,
    component_names: List[str],
    prompt_key: str,
    prompt_path: str,
) -> None:
    """Add a prompt path to the specified section of the configuration."""
    matches = [
        component
        for component in config.get(section_key, [])
        if component.get(CONFIG_NAME_KEY) in component_names
    ]

    if not matches:
        return

    # Update the first occurrence of the component.
    matches[0][prompt_key] = prompt_path

    if len(matches) > 1:
        structlogger.warning(
            "rasa.studio.prompts.add_prompt_to_config.multiple_components",
            event_info=(
                "Multiple components found in the configuration for the same prompt."
            ),
        )
