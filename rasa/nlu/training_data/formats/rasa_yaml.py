import logging
from typing import Text, Any, List, Dict, Tuple

import typing

from rasa.nlu.training_data.entities_parser import EntitiesParser
from rasa.nlu.training_data.formats.readerwriter import TrainingDataReader
import rasa.utils.io
from rasa.nlu.training_data.lookup_tables_parser import LookupTablesParser
from rasa.nlu.training_data.synonyms_parser import SynonymsParser

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import TrainingData


logger = logging.getLogger(__name__)

KEY_NLU = "nlu"
KEY_INTENT = "intent"
KEY_INTENT_EXAMPLES = "examples"
KEY_INTENT_TEXT = "text"
KEY_SYNONYM = "synonym"
KEY_SYNONYM_EXAMPLES = "examples"
KEY_REGEX = "regex"
KEY_REGEX_EXAMPLES = "examples"
KEY_LOOKUP = "lookup"
KEY_LOOKUP_EXAMPLES = "examples"

DOCS_LINK = "https://rasa.com/docs/rasa/"


class RasaYAMLReader(TrainingDataReader):
    def __init__(self) -> None:
        self.training_examples = []
        self.entity_synonyms = {}
        self.regex_features = []
        self.lookup_tables = []

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        from rasa.nlu.training_data import TrainingData

        self.__init__()

        yaml_content = rasa.utils.io.read_yaml(s)

        for key, value in yaml_content.items():
            if key == KEY_NLU:
                self._parse_nlu(value)
            else:
                logger.warning(
                    f"Unexpected key '{key}' found in {self.filename}"
                    f"Acceptable keys are: {KEY_NLU}"
                )

        return TrainingData(
            self.training_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
        )

    def _parse_nlu(self, nlu_data: List[Dict[Text, Any]]) -> None:

        for nlu_item in nlu_data:
            if not isinstance(nlu_item, dict):
                logger.warning(
                    f"Unexpected block found in '{self.filename}': \n"
                    f">> {nlu_item}\n"
                    f"Items under the `nlu` key must be YAML dictionaries."
                    f"This block will be skipped."
                    f"Please check the docs at {DOCS_LINK}."
                )
                continue

            if KEY_INTENT in nlu_item.keys():
                self._parse_intent(nlu_item)
            elif KEY_SYNONYM in nlu_item.keys():
                self._parse_synonym(nlu_item)
            elif KEY_REGEX in nlu_item.keys():
                self._parse_regex(nlu_item)
            elif KEY_LOOKUP in nlu_item.keys():
                self._parse_lookup(nlu_item)

    def _parse_intent(self, data: Dict[Text, Any]) -> None:
        from rasa.nlu.training_data import Message

        intent = data.get(KEY_INTENT, "")
        examples = data.get(KEY_INTENT_EXAMPLES, "")

        for example, entities in self._parse_training_examples(examples):

            SynonymsParser.add_synonyms_from_entities(
                example, entities, self.entity_synonyms
            )

            plain_text = EntitiesParser.replace_entities(example)

            message = Message.build(plain_text, intent)
            message.set("entities", entities)
            self.training_examples.append(message)

    def _parse_training_examples(self, examples: Text) -> List[Tuple[Text, List[Dict]]]:

        if isinstance(examples, list):
            iterable = [
                example.get(KEY_INTENT_TEXT, "") for example in examples if example
            ]
        elif isinstance(examples, str):
            iterable = examples.splitlines()
        else:
            logger.warning(
                f"Unexpected block found in '{self.filename}':\n"
                f">> {examples}\n"
                f"This block will be skipped."
                f"Please check the docs at {DOCS_LINK}."
            )
            return []

        results = []
        for example in iterable:
            entities = EntitiesParser.find_entities_in_training_example(example)
            results.append((example, entities))

        return results

    def _parse_synonym(self, nlu_item: Dict[Text, Any]) -> None:

        synonym_name = nlu_item[KEY_SYNONYM]
        examples = nlu_item.get(KEY_SYNONYM_EXAMPLES, "")

        if not examples:
            logger.warning(f"{KEY_SYNONYM}: {synonym_name} doesn't have any examples.")

        if not isinstance(examples, str):
            logger.warning(
                f"Unexpected block found in {self.filename}:\n"
                f">> {examples}\n"
                f"It will be skipped."
                f"This block will be skipped."
                f"Please check the docs at {DOCS_LINK}."
            )
            return

        for example in examples.splitlines():
            SynonymsParser.add_synonym(example, synonym_name, self.entity_synonyms)

    def _parse_regex(self, nlu_item: Dict[Text, Any]) -> None:

        regex_name = nlu_item[KEY_REGEX]
        examples = nlu_item.get(KEY_REGEX_EXAMPLES, "")

        if not examples:
            logger.warning(f"{KEY_REGEX}: {regex_name} doesn't have any examples.")

        if not isinstance(examples, str):
            logger.warning(
                f"Unexpected block found in {self.filename}:\n"
                f">> {examples}\n"
                f"This block will be skipped."
                f"Please check the docs at {DOCS_LINK}."
            )
            return

        for example in examples.splitlines():
            self.regex_features.append({"name": regex_name, "pattern": example})

    def _parse_lookup(self, nlu_item: Dict[Text, Any]):

        lookup_item_name = nlu_item[KEY_LOOKUP]
        examples = nlu_item.get(KEY_LOOKUP_EXAMPLES, "")

        if not examples:
            logger.warning(
                f"{KEY_LOOKUP}: {lookup_item_name} doesn't have any examples."
            )

        if not isinstance(examples, str):
            logger.warning(
                f"Unexpected block found in {self.filename}:\n"
                f">> {examples}\n"
                f"This block will be skipped."
                f"Please check the docs at {DOCS_LINK}."
            )
            return

        for example in examples.splitlines():
            LookupTablesParser.add_item_to_lookup_tables(
                lookup_item_name, example, self.lookup_tables
            )
