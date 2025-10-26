from typing import Any, Dict, Literal, Optional, Text

import structlog

from rasa.core.config.configuration import Configuration
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_NAME_CLAUDE_3_5_SONNET_20240620,
    MODEL_NAME_GPT_4O_2024_11_20,
    OPENAI_PROVIDER,
    PROVIDER_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.dialogue_understanding.generator.single_step.single_step_based_llm_command_generator import (  # noqa: E501
    SingleStepBasedLLMCommandGenerator,
)
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    ANTHROPIC_PROVIDER,
    AWS_BEDROCK_PROVIDER,
    AZURE_OPENAI_PROVIDER,
    MAX_COMPLETION_TOKENS_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    TEMPERATURE_CONFIG_KEY,
)
from rasa.shared.utils.llm import (
    get_default_prompt_template_based_on_model,
    get_prompt_template,
)

structlogger = structlog.get_logger()


DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: MODEL_NAME_GPT_4O_2024_11_20,
    TEMPERATURE_CONFIG_KEY: 0.0,
    MAX_COMPLETION_TOKENS_CONFIG_KEY: DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    TIMEOUT_CONFIG_KEY: 7,
}

# Non-agent default prompt mapping (used by default)
MODEL_PROMPT_MAPPER = {
    f"{OPENAI_PROVIDER}/{MODEL_NAME_GPT_4O_2024_11_20}": (
        "command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
    ),
    f"{AZURE_OPENAI_PROVIDER}/{MODEL_NAME_GPT_4O_2024_11_20}": (
        "command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
    ),
    f"{AWS_BEDROCK_PROVIDER}/anthropic.{MODEL_NAME_CLAUDE_3_5_SONNET_20240620}-v1:0": (
        "command_prompt_v2_claude_3_5_sonnet_20240620_template.jinja2"
    ),
    f"{ANTHROPIC_PROVIDER}/{MODEL_NAME_CLAUDE_3_5_SONNET_20240620}": (
        "command_prompt_v2_claude_3_5_sonnet_20240620_template.jinja2"
    ),
}

# Agentic prompt mapping (used only when agents are configured)
AGENT_MODEL_PROMPT_MAPPER = {
    f"{OPENAI_PROVIDER}/{MODEL_NAME_GPT_4O_2024_11_20}": (
        "agent_command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
    ),
    f"{AZURE_OPENAI_PROVIDER}/{MODEL_NAME_GPT_4O_2024_11_20}": (
        "agent_command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
    ),
    f"{AWS_BEDROCK_PROVIDER}/anthropic.{MODEL_NAME_CLAUDE_3_5_SONNET_20240620}-v1:0": (
        "agent_command_prompt_v2_claude_3_5_sonnet_20240620_template.jinja2"
    ),
    f"{ANTHROPIC_PROVIDER}/{MODEL_NAME_CLAUDE_3_5_SONNET_20240620}": (
        "agent_command_prompt_v2_claude_3_5_sonnet_20240620_template.jinja2"
    ),
}

# Defaults for non-agent prompts
DEFAULT_COMMAND_PROMPT_TEMPLATE_FILE_NAME = (
    "command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
)
FALLBACK_COMMAND_PROMPT_TEMPLATE_FILE_NAME = (
    "command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
)

# Defaults for agentic prompts
AGENT_DEFAULT_COMMAND_PROMPT_TEMPLATE_FILE_NAME = (
    "agent_command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
)
AGENT_FALLBACK_COMMAND_PROMPT_TEMPLATE_FILE_NAME = (
    "agent_command_prompt_v2_gpt_4o_2024_11_20_template.jinja2"
)


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class CompactLLMCommandGenerator(SingleStepBasedLLMCommandGenerator):
    """A single step LLM-based command generator."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config,
            model_storage,
            resource,
            prompt_template=prompt_template,
            **kwargs,
        )

    @staticmethod
    def get_default_prompt_template_file_name() -> str:
        """Get the default prompt template file name for the command generator."""
        return (
            AGENT_DEFAULT_COMMAND_PROMPT_TEMPLATE_FILE_NAME
            if Configuration.get_instance().available_agents.has_agents()
            else DEFAULT_COMMAND_PROMPT_TEMPLATE_FILE_NAME
        )

    @staticmethod
    def get_fallback_prompt_template_file_name() -> str:
        """Get the fallback prompt template file name for the command generator."""
        return (
            AGENT_FALLBACK_COMMAND_PROMPT_TEMPLATE_FILE_NAME
            if Configuration.get_instance().available_agents.has_agents()
            else FALLBACK_COMMAND_PROMPT_TEMPLATE_FILE_NAME
        )

    @staticmethod
    def get_model_prompt_mapper() -> Dict[str, str]:
        """Get the model prompt mapper for the command generator."""
        return (
            AGENT_MODEL_PROMPT_MAPPER
            if Configuration.get_instance().available_agents.has_agents()
            else MODEL_PROMPT_MAPPER
        )

    @staticmethod
    def get_default_llm_config() -> Dict[str, Any]:
        """Get the default LLM config for the command generator."""
        return DEFAULT_LLM_CONFIG

    @staticmethod
    def get_component_command_syntax_version() -> CommandSyntaxVersion:
        return CommandSyntaxVersion.v2

    @classmethod
    def _resolve_component_prompt_template(
        cls,
        config: Dict[str, Any],
        prompt_template: Optional[str] = None,
        log_context: Optional[Literal["init", "fingerprint_addon"]] = None,
        log_source_component: Optional[str] = "CompactLLMCommandGenerator",
    ) -> Optional[str]:
        """Get the prompt template from the config or the default prompt template."""
        if prompt_template is not None:
            return prompt_template

        # Try to load the template from the given path or fallback to the default for
        # the component.
        custom_prompt_template_path = config.get(PROMPT_TEMPLATE_CONFIG_KEY)
        if custom_prompt_template_path is not None:
            custom_prompt_template = get_prompt_template(
                custom_prompt_template_path,
                None,  # Default will be based on the model
                log_source_component=log_source_component,
                log_source_method=log_context,
            )
            if custom_prompt_template is not None:
                return custom_prompt_template

        # Fallback to the default prompt template based on the model.
        default_command_prompt_template = get_default_prompt_template_based_on_model(
            llm_config=config.get(LLM_CONFIG_KEY, {}) or {},
            model_prompt_mapping=cls.get_model_prompt_mapper(),
            default_prompt_path=cls.get_default_prompt_template_file_name(),
            fallback_prompt_path=cls.get_fallback_prompt_template_file_name(),
            log_source_component=log_source_component,
            log_source_method=log_context,
        )

        return default_command_prompt_template
