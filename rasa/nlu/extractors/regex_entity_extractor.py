from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Text

import rasa.nlu.utils.pattern_utils as pattern_utils
import rasa.shared.utils.io
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR, is_trainable=True
)
class RegexEntityExtractor(GraphComponent, EntityExtractorMixin):
    """Extracts entities via lookup tables and regexes defined in the training data."""

    REGEX_FILE_NAME = "regex.json"

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            # text will be processed with case insensitive as default
            "case_sensitive": False,
            # use lookup tables to extract entities
            "use_lookup_tables": True,
            # use regexes to extract entities
            "use_regexes": True,
            # use match word boundaries for lookup table
            "use_word_boundaries": True,
        }

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> RegexEntityExtractor:
        """Creates a new `GraphComponent`.

        Args:
            config: This config overrides the `default_config`.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            execution_context: Information about the current graph run. Unused.

        Returns: An instantiated `GraphComponent`.
        """
        return cls(config, model_storage, resource)

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        patterns: Optional[List[Dict[Text, Text]]] = None,
    ) -> None:
        """Creates a new instance.

        Args:
            config: The configuration.
            model_storage: Storage which graph components can use to persist and load
                themselves.
            resource: Resource locator for this component which can be used to persist
                and load itself from the `model_storage`.
            patterns: a list of patterns
        """
        # graph component
        self._config = {**self.get_default_config(), **config}
        self._model_storage = model_storage
        self._resource = resource
        # extractor
        self.case_sensitive = self._config["case_sensitive"]
        self.patterns = patterns or []

    def train(self, training_data: TrainingData) -> Resource:
        """Extract patterns from the training data.

        Args:
            training_data: the training data
        """
        self.patterns = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self._config["use_lookup_tables"],
            use_regexes=self._config["use_regexes"],
            use_only_entities=True,
            use_word_boundaries=self._config["use_word_boundaries"],
        )

        if not self.patterns:
            rasa.shared.utils.io.raise_warning(
                "No lookup tables or regexes defined in the training data that have "
                "a name equal to any entity in the training data. In order for this "
                "component to work you need to define valid lookup tables or regexes "
                "in the training data."
            )
        self.persist()
        return self._resource

    def process(self, messages: List[Message]) -> List[Message]:
        """Extracts entities from messages and appends them to the attribute.

        If no patterns where found during training, then the given messages will not
        be modified. In particular, if no `ENTITIES` attribute exists yet, then
        it will *not* be created.

        If no pattern can be found in the given message, then no entities will be
        added to any existing list of entities. However, if no `ENTITIES` attribute
        exists yet, then an `ENTITIES` attribute will be created.

        Returns:
           the given list of messages that have been modified
        """
        if not self.patterns:
            rasa.shared.utils.io.raise_warning(
                f"The {self.__class__.__name__} has not been "
                f"trained properly yet. "
                f"Continuing without extracting entities via this extractor."
            )
            return messages

        for message in messages:
            extracted_entities = self._extract_entities(message)
            extracted_entities = self.add_extractor_name(extracted_entities)
            message.set(
                ENTITIES,
                message.get(ENTITIES, []) + extracted_entities,
                add_to_output=True,
            )
        return messages

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message.

        Args:
            message: a message
        Returns:
            a list of dictionaries describing the entities
        """
        entities = []

        flags = 0  # default flag
        if not self.case_sensitive:
            flags = re.IGNORECASE

        for pattern in self.patterns:
            matches = re.finditer(pattern["pattern"], message.get(TEXT), flags=flags)

            for match in matches:
                start_index = match.start()
                end_index = match.end()
                entities.append(
                    {
                        ENTITY_ATTRIBUTE_TYPE: pattern["name"],
                        ENTITY_ATTRIBUTE_START: start_index,
                        ENTITY_ATTRIBUTE_END: end_index,
                        ENTITY_ATTRIBUTE_VALUE: message.get(TEXT)[
                            start_index:end_index
                        ],
                    }
                )

        return entities

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> RegexEntityExtractor:
        """Loads trained component (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_path:
                regex_file = model_path / cls.REGEX_FILE_NAME
                patterns = rasa.shared.utils.io.read_json_file(regex_file)
                return cls(
                    config,
                    model_storage=model_storage,
                    resource=resource,
                    patterns=patterns,
                )
        except (ValueError, FileNotFoundError):
            rasa.shared.utils.io.raise_warning(
                f"Failed to load {cls.__name__} from model storage. "
                f"This can happen if the model could not be trained because regexes "
                f"could not be extracted from the given training data - and hence "
                f"could not be persisted."
            )
            return cls(config, model_storage=model_storage, resource=resource)

    def persist(self) -> None:
        """Persist this model."""
        if not self.patterns:
            return
        with self._model_storage.write_to(self._resource) as model_path:
            regex_file = model_path / self.REGEX_FILE_NAME
            rasa.shared.utils.io.dump_obj_as_json_to_file(regex_file, self.patterns)
