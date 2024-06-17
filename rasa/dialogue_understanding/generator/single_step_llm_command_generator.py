from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.dialogue_understanding.generator.llm_command_generator import (
    LLMCommandGenerator,
)


@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.COMMAND_GENERATOR,
    ],
    is_trainable=True,
)
class SingleStepLLMCommandGenerator(LLMCommandGenerator):
    pass
