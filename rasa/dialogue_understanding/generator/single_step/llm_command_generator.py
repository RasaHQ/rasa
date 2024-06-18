import structlog
from typing import Dict, Any, Optional, Text
from deprecated import deprecated  # type: ignore[import]


from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.dialogue_understanding.generator.single_step.single_step_llm_command_generator import (  # noqa: E501
    SingleStepLLMCommandGenerator,
)
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.utils.io import raise_deprecation_warning

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
        "Please use use SingleStepLLMCommandGenerator instead."
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
            message="LLMCommandGenerator is deprecated and will be  "
            "removed in Rasa 4.0.0. "
            "Please use SingleStepLLMCommandGenerator instead."
        )

        super().__init__(
            config,
            model_storage,
            resource,
            prompt_template=prompt_template,
            **kwargs,
        )
