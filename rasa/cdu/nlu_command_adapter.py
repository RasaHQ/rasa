import structlog

from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

structlogger = structlog.get_logger()

# TODO: check if the original inhertance from IntentClassifier and EntityExtractorMixin
#   is still needed or what benefits that provided.
@DefaultV1Recipe.register(
    [
        DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER,
        DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    ],
    is_trainable=False,
)
class NLUCommandAdapter(GraphComponent):
    # TODO: implement
    pass
