import os
from typing import Any, Dict, List, Optional, Text

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.shared.constants import DOCS_URL_TRAINING_DATA
from rasa.shared.nlu.constants import ENTITIES, TEXT
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.utils import write_json_to_file
from rasa.nlu.extractors.extractor import EntityExtractorMixin
import rasa.utils.io
import rasa.shared.utils.io
from rasa.nlu.extractors._entity_synonyms import EntitySynonymMapper
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage

# This is a workaround around until we have all components migrated to `GraphComponent`.
EntitySynonymMapper = EntitySynonymMapper


class EntitySynonymMapperComponent(GraphComponent, EntityExtractorMixin):
    """Maps entities to their synonyms if they appear in the training data."""

    SYNONYM_FILENAME = "synonyms.json"

    def __init__(
        self,
        config: Optional[Dict[Text, Any]] = None,
        model_storage: ModelStorage = None,
        resource: Resource = None,
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

        self._synonyms = synonyms if synonyms else {}

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> EntitySynonymMapperComponent:
        """Creates component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, synonyms)

    def train(self, training_data: TrainingData,) -> None:
        """Trains the synonym lookup table."""
        for key, value in list(training_data.entity_synonyms.items()):
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get(ENTITIES, []):
                entity_val = example.get(TEXT)[entity["start"] : entity["end"]]
                self.add_entities_if_synonyms(entity_val, str(entity.get("value")))

        if self._resource:
            self._persist()

    def process(self, messages: List[Message]) -> None:
        """Modifies entities attached to message to resolve synonyms."""
        for message in messages:
            updated_entities = message.get(ENTITIES, [])[:]
            self.replace_synonyms(updated_entities)
            message.set(ENTITIES, updated_entities, add_to_output=True)

    def _persist(self) -> None:

        if self._synonyms:
            with self._model_storage.write_to(self._resource) as storage:
                entity_synonyms_file = os.path.join(
                    storage, EntitySynonymMapperComponent.SYNONYM_FILENAME
                )
                write_json_to_file(
                    entity_synonyms_file, self._synonyms, separators=(",", ": ")
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
    ) -> EntitySynonymMapperComponent:
        """Loads trained component (see parent class for full docstring)."""
        with model_storage.read_from(resource) as storage:
            entity_synonyms_file = os.path.join(
                storage, EntitySynonymMapperComponent.SYNONYM_FILENAME
            )
            if os.path.isfile(entity_synonyms_file):
                synonyms = rasa.shared.utils.io.read_json_file(entity_synonyms_file)
            else:
                synonyms = None
                rasa.shared.utils.io.raise_warning(
                    f"Failed to load synonyms file from '{entity_synonyms_file}'.",
                    docs=DOCS_URL_TRAINING_DATA + "#synonyms",
                )
        return cls(config, model_storage, resource, synonyms)

    def replace_synonyms(self, entities: List[Dict[Text, Any]]) -> None:
        """Replace any entities which match a synonym with the synonymous entity."""
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity["value"])
            if entity_value.lower() in self._synonyms:
                entity["value"] = self._synonyms[entity_value.lower()]
                self.add_processor_name(entity)

    def _add_entities_if_synonyms(
        self, entity_a: Text, entity_b: Optional[Text]
    ) -> None:
        if entity_b is not None:
            original = str(entity_a)
            replacement = str(entity_b)

            if original != replacement:
                original = original.lower()
                if (
                    original in self._synonyms
                    and self._synonyms[original] != replacement
                ):
                    rasa.shared.utils.io.raise_warning(
                        f"Found conflicting synonym definitions "
                        f"for {repr(original)}. Overwriting target "
                        f"{repr(self._synonyms[original])} with "
                        f"{repr(replacement)}. "
                        f"Check your training data and remove "
                        f"conflicting synonym definitions to "
                        f"prevent this from happening.",
                        docs=DOCS_URL_TRAINING_DATA + "#synonyms",
                    )

                self._synonyms[original] = replacement
