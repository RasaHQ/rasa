import os
from typing import Any, Dict, List, Optional, Text, Type

from rasa.nlu.components import Component
from rasa.shared.constants import DOCS_URL_TRAINING_DATA
from rasa.shared.nlu.constants import (
    ENTITIES,
    TEXT,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_VALUE,
)
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.training_data import (
    TrainingDataFull,
    TrainingDataChunk,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.utils import write_json_to_file
import rasa.utils.io
import rasa.shared.utils.io
from rasa.utils.tensorflow.data_generator import DataChunkFile


class EntitySynonymMapper(EntityExtractor):
    """Maps synonymous entity values to the same value."""

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specifies which components need to be present in the pipeline."""
        return [EntityExtractor]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        synonyms: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Initializes the entity synonym mapper.

        Args:
            component_config: The component configuration.
            synonyms: The synonyms to use.
        """
        super().__init__(component_config)

        self.synonyms = synonyms if synonyms else {}

    def prepare_partial_training(
        self,
        training_data: TrainingDataFull,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Prepare the component for training on just a part of the data.

        See parent class for more information.
        """
        self._add_synonyms_from_data(training_data)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component."""
        self._add_synonyms_from_data(training_data)
        self._process_entity_examples(training_data.entity_synonyms)

    def train_on_chunks(
        self,
        data_chunk_files: List[DataChunkFile],
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Trains this component using the list of data chunk files.

        See parent class for more information.
        """
        for data_chunk in data_chunk_files:
            training_data_chunk = TrainingDataChunk.load_chunk(data_chunk.file_path)
            self._process_entity_examples(training_data_chunk.entity_examples)

    def _add_synonyms_from_data(self, training_data: TrainingData) -> None:
        """Adds synonyms from data to the list of synonyms."""
        """Prepare the component for training on just a part of the data.

        See parent class for more information.
        """
        for key, value in list(training_data.entity_synonyms.items()):
            self._add_entities_if_synonyms(key, value)

    def _process_entity_examples(self, entity_examples: List[Message]) -> None:
        for example in entity_examples:
            for entity in example.get(ENTITIES, []):
                entity_val = example.get(TEXT)[
                    entity[ENTITY_ATTRIBUTE_START] : entity[ENTITY_ATTRIBUTE_END]
                ]
                self._add_entities_if_synonyms(
                    entity_val, str(entity.get(ENTITY_ATTRIBUTE_VALUE))
                )

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message."""
        updated_entities = message.get(ENTITIES, [])[:]
        self._replace_synonyms(updated_entities)
        message.set(ENTITIES, updated_entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""
        if self.synonyms:
            file_name = file_name + ".json"
            entity_synonyms_file = os.path.join(model_dir, file_name)
            write_json_to_file(
                entity_synonyms_file, self.synonyms, separators=(",", ": ")
            )
            return {"file": file_name}
        else:
            return {"file": None}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["EntitySynonymMapper"] = None,
        **kwargs: Any,
    ) -> "EntitySynonymMapper":

        file_name = meta.get("file")
        if not file_name:
            synonyms = None
            return cls(meta, synonyms)

        entity_synonyms_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entity_synonyms_file):
            synonyms = rasa.shared.utils.io.read_json_file(entity_synonyms_file)
        else:
            synonyms = None
            rasa.shared.utils.io.raise_warning(
                f"Failed to load synonyms file from '{entity_synonyms_file}'.",
                docs=DOCS_URL_TRAINING_DATA + "#synonyms",
            )
        return cls(meta, synonyms=synonyms)

    def _replace_synonyms(self, entities: List[Dict[Text, Any]]) -> None:
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity[ENTITY_ATTRIBUTE_VALUE])
            if entity_value.lower() in self.synonyms:
                entity[ENTITY_ATTRIBUTE_VALUE] = self.synonyms[entity_value.lower()]
                self.add_processor_name(entity)

    def _add_entities_if_synonyms(
        self, entity_a: Optional[Text], entity_b: Optional[Text]
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
