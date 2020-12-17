import logging
import os
import re
import typing
from typing import Any, Dict, List, Optional, Text

import rasa.shared.utils.io
import rasa.nlu.utils.pattern_utils as pattern_utils
from rasa.nlu.components import UnsupportedLanguageError
from rasa.nlu.model import Metadata
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import (
    ENTITIES,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    TEXT,
    ENTITY_ATTRIBUTE_TYPE,
)
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.shared.core.domain import Domain

if typing.TYPE_CHECKING:
    from rasa.nlu.components import Component


logger = logging.getLogger(__name__)


class RegexEntityExtractor(EntityExtractor):
    """Searches for entities in the user's message using the lookup tables and regexes
    defined in the training data."""

    defaults = {
        # text will be processed with case insensitive as default
        "case_sensitive": False,
        # use lookup tables to extract entities
        "use_lookup_tables": True,
        # use regexes to extract entities
        "use_regexes": True,
        # use match word boundaries for lookup table
        "use_word_boundaries": True,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        patterns: Optional[List[Dict[Text, Text]]] = None,
    ):
        """Extracts entities using the lookup tables and/or regexes defined in the training data."""
        super(RegexEntityExtractor, self).__init__(component_config)

        self.case_sensitive = self.component_config["case_sensitive"]
        self.patterns = patterns or []
        self.entity_names = None

    @classmethod
    def create(
        cls,
        component_config: Dict[Text, Any],
        model_config: RasaNLUModelConfig,
        domain: Optional[Domain] = None,
    ) -> "Component":
        """Creates this component (e.g. before a training is started).

        Method can access all configuration parameters.

        Args:
            component_config: The components configuration parameters.
            model_config: The model configuration parameters.
            domain: The domain the model uses.

        Returns:
            The created component.
        """
        # Check language supporting
        language = model_config.language
        if not cls.can_handle_language(language):
            # check failed
            raise UnsupportedLanguageError(cls.name, language)

        component = cls(component_config)

        if domain is not None:
            component.entity_names = domain.entities

        return component

    def prepare_partial_training(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Prepare the component for training on just a part of the data.

        See parent class for more information.
        """
        if self.entity_names is None:
            self.entity_names = training_data.entities

        self.patterns = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self.component_config["use_lookup_tables"],
            use_regexes=self.component_config["use_regexes"],
            patter_names=self.entity_names,
            use_word_boundaries=self.component_config["use_word_boundaries"],
        )

        if not self.patterns:
            rasa.shared.utils.io.raise_warning(
                "No lookup tables or regexes defined in the training data that have "
                "a name equal to any entity in the training data. In order for this "
                "component to work you need to define valid lookup tables or regexes "
                "in the training data."
            )

    def process(self, message: Message, **kwargs: Any) -> None:
        if not self.patterns:
            return

        extracted_entities = self._extract_entities(message)
        extracted_entities = self.add_extractor_name(extracted_entities)

        message.set(
            ENTITIES, message.get(ENTITIES, []) + extracted_entities, add_to_output=True
        )

    def _extract_entities(self, message: Message) -> List[Dict[Text, Any]]:
        """Extract entities of the given type from the given user message."""
        entities = []

        flags = 0  # default flag
        if not self.case_sensitive:
            flags = re.IGNORECASE

        for pattern in self.patterns:
            matches = re.finditer(pattern["pattern"], message.get(TEXT), flags=flags)
            matches = list(matches)

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
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["RegexEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "RegexEntityExtractor":

        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            patterns = rasa.shared.utils.io.read_json_file(regex_file)
            return RegexEntityExtractor(meta, patterns=patterns)

        return RegexEntityExtractor(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        file_name = f"{file_name}.json"
        regex_file = os.path.join(model_dir, file_name)
        rasa.shared.utils.io.dump_obj_as_json_to_file(regex_file, self.patterns)

        return {"file": file_name}
