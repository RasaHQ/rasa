from typing import Any, Dict, Optional, Text

import structlog
from deprecated import deprecated  # type: ignore[import-untyped]

from rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator import (  # noqa: E501
    SingleStepLLMCommandGenerator,
)
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.io import raise_deprecation_warning
from rasa.shared.utils.llm import LLMInput

structlogger = structlog.get_logger()


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
@deprecated(
    reason=(
        "The LLMCommandGenerator is deprecated and will be removed in Rasa 4.0.0. "
        "Please use SingleStepLLMCommandGenerator instead."
    )
)
class LLMCommandGenerator(SingleStepLLMCommandGenerator):
    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
        prompt_template: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        raise_deprecation_warning(
            message="`LLMCommandGenerator` has been renamed "
            "to `SingleStepLLMCommandGenerator`."
            "Support for the former name `LLMCommandGenerator` "
            "will be removed in Rasa `4.0.0`."
            "Please modify your assistant's configuration to "
            "use `SingleStepLLMCommandGenerator` instead."
        )

        super().__init__(
            config,
            model_storage,
            resource,
            prompt_template=prompt_template,
            **kwargs,
        )

    async def invoke_llm(self, llm_input: LLMInput) -> Optional[LLMResponse]:
        try:
            return await super().invoke_llm(llm_input)
        except ProviderClientAPIException:
            # Returning 'None' in case of a ProviderClientAPIException ensures
            # backward compatibility with previous versions. In earlier
            # implementations, 'invoke_llm' was expected to return 'None' when
            # encountering API-related issues. This practice is maintained here
            # to prevent breaking changes in how exceptions are handled and
            # propagated in existing integrations, where callers might not yet
            # be prepared to manage exceptions directly.
            return None
