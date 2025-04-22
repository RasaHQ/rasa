import importlib.resources
from typing import Any, Dict, Literal, Optional, Text

import structlog

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_LLM_CONFIG,
    FLOW_RETRIEVAL_KEY,
    LLM_CONFIG_KEY,
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
    EMBEDDINGS_CONFIG_KEY,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
)
from rasa.shared.utils.constants import LOG_COMPONENT_SOURCE_METHOD_FINGERPRINT_ADDON
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    get_prompt_template,
    resolve_model_client_config,
)

DEFAULT_COMMAND_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator.prompt_templates",
    "command_prompt_template.jinja2",
)


structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class SingleStepLLMCommandGenerator(CompactLLMCommandGenerator):
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

        # Set the prompt template
        if config.get(PROMPT_CONFIG_KEY):
            structlogger.warning(
                "single_step_llm_command_generator.init",
                event_info=(
                    "The config parameter 'prompt' is deprecated "
                    "and will be removed in Rasa 4.0.0. "
                    "Please use the config parameter 'prompt_template' instead. "
                ),
            )

        # Set the command syntax version to v1
        CommandSyntaxManager.set_syntax_version(
            self.get_component_command_syntax_version()
        )

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            PROMPT_CONFIG_KEY: None,  # Legacy
            PROMPT_TEMPLATE_CONFIG_KEY: None,
            USER_INPUT_CONFIG_KEY: None,
            LLM_CONFIG_KEY: None,
            FLOW_RETRIEVAL_KEY: FlowRetrieval.get_default_config(),
        }

    @classmethod
    def fingerprint_addon(cls: Any, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint for the graph."""
        prompt_template = cls._resolve_component_prompt_template(
            config, log_context=LOG_COMPONENT_SOURCE_METHOD_FINGERPRINT_ADDON
        )
        llm_config = resolve_model_client_config(
            config.get(LLM_CONFIG_KEY), SingleStepLLMCommandGenerator.__name__
        )
        embedding_config = resolve_model_client_config(
            config.get(FLOW_RETRIEVAL_KEY, {}).get(EMBEDDINGS_CONFIG_KEY),
            FlowRetrieval.__name__,
        )
        return deep_container_fingerprint(
            [prompt_template, llm_config, embedding_config]
        )

    @staticmethod
    def get_default_llm_config() -> Dict[str, Any]:
        """Get the default LLM config for the command generator."""
        return DEFAULT_LLM_CONFIG

    @staticmethod
    def get_component_command_syntax_version() -> CommandSyntaxVersion:
        return CommandSyntaxVersion.v1

    @staticmethod
    def _resolve_component_prompt_template(
        config: Dict[str, Any],
        prompt_template: Optional[str] = None,
        log_context: Optional[Literal["init", "fingerprint_addon"]] = None,
    ) -> Optional[str]:
        """Get the prompt template from the config or the default prompt template."""
        # Case when model is being loaded
        if prompt_template is not None:
            return prompt_template

        # The prompt can be configured in the config via the "prompt" (deprecated) or
        # "prompt_template" properties
        prompt_template_path = (
            config.get(PROMPT_CONFIG_KEY)
            or config.get(PROMPT_TEMPLATE_CONFIG_KEY)
            or None
        )

        # Try to load the template from the given path or fallback to the default for
        # the component
        return get_prompt_template(
            prompt_template_path,
            DEFAULT_COMMAND_PROMPT_TEMPLATE,
            log_source_component=SingleStepLLMCommandGenerator.__name__,
            log_source_method=log_context,
        )
