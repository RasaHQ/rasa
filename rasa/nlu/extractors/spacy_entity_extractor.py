import typing
from typing import Any, Dict, List, Text, Type

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, TEXT
from rasa.nlu.utils.spacy_utils import SpacyModel, SpacyNLP
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.training_data.message import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR,
    is_trainable=False,
    model_from="SpacyNLP",
)
class SpacyEntityExtractor(GraphComponent, EntityExtractorMixin):
    """Entity extractor which uses SpaCy."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [SpacyNLP]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # by default all dimensions recognized by spacy are returned
            # dimensions can be configured to contain an array of strings
            # with the names of the dimensions to filter for
            "dimensions": None
        }

    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize SpacyEntityExtractor."""
        self._config = config

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config)

    @staticmethod
    def required_packages() -> List[Text]:
        """Lists required dependencies (see parent class for full docstring)."""
        return ["spacy"]

    def process(self, messages: List[Message], model: SpacyModel) -> List[Message]:
        """Extract entities using SpaCy.

        Args:
            messages: List of messages to process.
            model: Container holding a loaded spacy nlp model.

        Returns: The processed messages.
        """
        for message in messages:
            # can't use the existing doc here (spacy_doc on the message)
            # because tokens are lower cased which is bad for NER
            spacy_nlp = model.model
            doc = spacy_nlp(message.get(TEXT))
            all_extracted = self.add_extractor_name(self._extract_entities(doc))
            dimensions = self._config["dimensions"]
            extracted = self.filter_irrelevant_entities(all_extracted, dimensions)
            message.set(
                ENTITIES, message.get(ENTITIES, []) + extracted, add_to_output=True
            )

        return messages

    @staticmethod
    def _extract_entities(doc: "Doc") -> List[Dict[Text, Any]]:
        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "confidence": None,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return entities
