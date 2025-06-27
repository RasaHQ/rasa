import importlib.resources
from typing import Any, Dict, Literal, Optional, Text

import structlog

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.generator.single_step.single_step_based_llm_command_generator import (  # noqa: E501
    SingleStepBasedLLMCommandGenerator,
)
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
)
from rasa.shared.utils.io import raise_deprecation_warning
from rasa.shared.utils.llm import (
    check_prompt_config_keys_and_warn_if_deprecated,
    get_prompt_template,
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
class SingleStepLLMCommandGenerator(SingleStepBasedLLMCommandGenerator):
    """A single step LLM-based command generator."""

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        raise_deprecation_warning(
            message=(
                "Support for `SingleStepLLMCommandGenerator` will be removed in Rasa "
                "`4.0.0`. Please modify your assistant's configuration to use the "
                "`CompactLLMCommandGenerator` or `SearchReadyLLMCommandGenerator` "
                "instead."
            )
        )
        super().__init__(
            config,
            model_storage,
            resource,
            prompt_template=prompt_template,
            **kwargs,
        )

        # Warn if the prompt config key is used to set the prompt template
        check_prompt_config_keys_and_warn_if_deprecated(
            config, "single_step_llm_command_generator"
        )

    @staticmethod
    def get_component_command_syntax_version() -> CommandSyntaxVersion:
        return CommandSyntaxVersion.v1

    @classmethod
    def _resolve_component_prompt_template(
        cls: Any,
        config: Dict[str, Any],
        prompt_template: Optional[str] = None,
        log_context: Optional[Literal["init", "fingerprint_addon"]] = None,
        log_source_component: Optional[str] = "SingleStepLLMCommandGenerator",
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
            log_source_component=log_source_component,
            log_source_method=log_context,
        )
