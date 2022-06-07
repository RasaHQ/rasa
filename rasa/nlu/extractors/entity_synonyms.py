from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Text
import logging

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.shared.constants import DOCS_URL_TRAINING_DATA
from rasa.shared.nlu.constants import ENTITIES, TEXT
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.utils import write_json_to_file
from rasa.nlu.extractors.extractor import EntityExtractorMixin
import rasa.utils.io
import rasa.shared.utils.io
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=True
)
class EntitySynonymMapper(GraphComponent, EntityExtractorMixin):
    """Maps entities to their synonyms if they appear in the training data."""

    SYNONYM_FILENAME = "synonyms.json"

    def __init__(
        self,
        config: Optional[Dict[Text, Any]],
        model_storage: ModelStorage,
        resource: Resource,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Creates the mapper.

        Args:
            config: The mapper's config.
            model_storage: Storage which the component can use to persist and load
                itself.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            synonyms: A dictionary of previously known synonyms.
        """
        self._config = config
        self._model_storage = model_storage
        self._resource = resource

        self.synonyms = synonyms if synonyms else {}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> EntitySynonymMapper:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, synonyms)

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the synonym lookup table."""
        for key, value in list(training_data.entity_synonyms.items()):
            self._add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get(ENTITIES, []):
                entity_val = example.get(TEXT)[entity["start"] : entity["end"]]
                self._add_entities_if_synonyms(entity_val, str(entity.get("value")))

        self._persist()
        return self._resource

    def process(self, messages: List[Message]) -> List[Message]:
        """Modifies entities attached to message to resolve synonyms.

        Args:
            messages: List containing the latest user message

        Returns:
            List containing the latest user message with entities resolved to
            synonyms if there is a match.
        """
        for message in messages:
            updated_entities = message.get(ENTITIES, [])[:]
            self.replace_synonyms(updated_entities)
            message.set(ENTITIES, updated_entities, add_to_output=True)

        return messages

    def _persist(self) -> None:

        if self.synonyms:
            with self._model_storage.write_to(self._resource) as storage:
                entity_synonyms_file = storage / EntitySynonymMapper.SYNONYM_FILENAME

                write_json_to_file(
                    entity_synonyms_file, self.synonyms, separators=(",", ": ")
                )

    # Adapt to get path from model storage and resource
    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> EntitySynonymMapper:
        """Loads trained component (see parent class for full docstring)."""
        synonyms = None
        try:
            with model_storage.read_from(resource) as storage:
                entity_synonyms_file = storage / EntitySynonymMapper.SYNONYM_FILENAME

                if os.path.isfile(entity_synonyms_file):
                    synonyms = rasa.shared.utils.io.read_json_file(entity_synonyms_file)
                else:
                    synonyms = None
                    rasa.shared.utils.io.raise_warning(
                        f"Failed to load synonyms file from '{entity_synonyms_file}'.",
                        docs=DOCS_URL_TRAINING_DATA + "#synonyms",
                    )
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )

        return cls(config, model_storage, resource, synonyms)

    def replace_synonyms(self, entities: List[Dict[Text, Any]]) -> None:
        """Replace any entities which match a synonym with the synonymous entity."""
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity["value"])
            if entity_value.lower() in self.synonyms:
                entity["value"] = self.synonyms[entity_value.lower()]
                self.add_processor_name(entity)

    def _add_entities_if_synonyms(
        self, entity_a: Text, entity_b: Optional[Text]
    ) -> None:
        if entity_b is not None:
            original = str(entity_a)
            replacement = str(entity_b)

            if original != replacement:
                original = original.lower()
                if original in self.synonyms and self.synonyms[original] != replacement:
                    rasa.shared.utils.io.raise_warning(
                        f"Found conflicting synonym definitions "
                        f"for {repr(original)}. Overwriting target "
                        f"{repr(self.synonyms[original])} with "
                        f"{repr(replacement)}. "
                        f"Check your training data and remove "
                        f"conflicting synonym definitions to "
                        f"prevent this from happening.",
                        docs=DOCS_URL_TRAINING_DATA + "#synonyms",
                    )

                self.synonyms[original] = replacement
