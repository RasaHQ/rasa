from typing import Any, Dict, Optional, Text

import structlog

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    FLOW_RETRIEVAL_KEY,
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_NAME_CLAUDE_3_5_SONNET_20240620,
    MODEL_NAME_GPT_4O_2024_11_20,
    OPENAI_PROVIDER,
    PROVIDER_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
    USER_INPUT_CONFIG_KEY,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.dialogue_understanding.generator.single_step.compact_llm_command_generator import (  # noqa: E501
    CompactLLMCommandGenerator,
)
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    ANTHROPIC_PROVIDER,
    AWS_BEDROCK_PROVIDER,
    AZURE_OPENAI_PROVIDER,
    MAX_TOKENS_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    TEMPERATURE_CONFIG_KEY,
)

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class SearchReadyLLMCommandGenerator(CompactLLMCommandGenerator):
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
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            PROMPT_TEMPLATE_CONFIG_KEY: None,
            USER_INPUT_CONFIG_KEY: None,
            LLM_CONFIG_KEY: None,
            FLOW_RETRIEVAL_KEY: FlowRetrieval.get_default_config(),
        }

    @staticmethod
    def get_default_llm_config() -> Dict[str, Any]:
        """Get the default LLM config for the command generator."""
        return {
            PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
            MODEL_CONFIG_KEY: MODEL_NAME_GPT_4O_2024_11_20,
            TEMPERATURE_CONFIG_KEY: 0.0,
            MAX_TOKENS_CONFIG_KEY: DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
            TIMEOUT_CONFIG_KEY: 7,
        }

    @staticmethod
    def get_default_prompt_template_file_name() -> str:
        """Get the default prompt template file name for the command generator."""
        return "command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"

    @staticmethod
    def get_fallback_prompt_template_file_name() -> str:
        """Get the fallback prompt template file name for the command generator."""
        return "command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"

    @staticmethod
    def get_model_prompt_mapper() -> Dict[str, str]:
        """Get the model prompt mapper for the command generator."""
        return {
            f"{OPENAI_PROVIDER}/{MODEL_NAME_GPT_4O_2024_11_20}": (
                "command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"
            ),
            f"{AZURE_OPENAI_PROVIDER}/{MODEL_NAME_GPT_4O_2024_11_20}": (
                "command_prompt_v3_gpt_4o_2024_11_20_template.jinja2"
            ),
            f"{AWS_BEDROCK_PROVIDER}/anthropic."
            f"{MODEL_NAME_CLAUDE_3_5_SONNET_20240620}-v1:0": (
                "command_prompt_v2_claude_3_5_sonnet_20240620_template.jinja2"
            ),
            f"{ANTHROPIC_PROVIDER}/{MODEL_NAME_CLAUDE_3_5_SONNET_20240620}": (
                "command_prompt_v2_claude_3_5_sonnet_20240620_template.jinja2"
            ),
        }

    @staticmethod
    def get_component_command_syntax_version() -> CommandSyntaxVersion:
        return CommandSyntaxVersion.v3
