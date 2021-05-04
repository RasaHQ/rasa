import os
from typing import Any, Dict, List, Optional, Text, Type

from rasa.architecture_prototype.persistence import ComponentPersistor
from rasa.nlu.components import Component
from rasa.shared.constants import DOCS_URL_TRAINING_DATA
from rasa.shared.nlu.constants import ENTITIES, TEXT
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.utils import write_json_to_file
from rasa.utils import train_utils
import rasa.utils.io
import rasa.shared.utils.io


class EntitySynonymMapper(EntityExtractor):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [EntityExtractor]

    def __init__(
        self,
        synonyms: Optional[Dict[Text, Any]] = None,
        persistor: Optional[ComponentPersistor] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(persistor=persistor, **kwargs)

        self.synonyms = synonyms if synonyms else {}

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> Optional[Text]:

        for key, value in list(training_data.entity_synonyms.items()):
            self.add_entities_if_synonyms(key, value)

        for example in training_data.entity_examples:
            for entity in example.get(ENTITIES, []):
                entity_val = example.get(TEXT)[entity["start"] : entity["end"]]
                self.add_entities_if_synonyms(entity_val, str(entity.get("value")))

        return self.persist()

    def process(self, messages: List[Message], **kwargs: Any) -> List[Message]:
        for message in messages:
            updated_entities = message.get(ENTITIES, [])[:]
            self.replace_synonyms(updated_entities)
            message.set(ENTITIES, updated_entities, add_to_output=True)

        return messages

    def persist(self,) -> Optional[Text]:

        if self.synonyms:
            entity_synonyms_file = self._persistor.file_for("synonyms.json")
            write_json_to_file(
                entity_synonyms_file, self.synonyms, separators=(",", ": ")
            )

        return self._persistor.resource_name()

    @classmethod
    def load(
        cls,
        persistor: ComponentPersistor,
        resource_name: Text,
        cached_component: Optional["EntitySynonymMapper"] = None,
        **kwargs: Any,
    ) -> "EntitySynonymMapper":
        """Loads trained component (see parent class for full docstring)."""
        entity_synonyms_file = persistor.get_resource(resource_name, "synonyms.json")
        if os.path.isfile(entity_synonyms_file):
            synonyms = rasa.shared.utils.io.read_json_file(entity_synonyms_file)
        else:
            synonyms = None
            rasa.shared.utils.io.raise_warning(
                f"Failed to load synonyms file from '{entity_synonyms_file}'.",
                docs=DOCS_URL_TRAINING_DATA + "#synonyms",
            )
        return cls(synonyms, persistor=persistor, **kwargs)

    def replace_synonyms(self, entities: List[Dict[Text, Any]]) -> None:
        for entity in entities:
            # need to wrap in `str` to handle e.g. entity values of type int
            entity_value = str(entity["value"])
            if entity_value.lower() in self.synonyms:
                entity["value"] = self.synonyms[entity_value.lower()]
                self.add_processor_name(entity)

    def add_entities_if_synonyms(
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
