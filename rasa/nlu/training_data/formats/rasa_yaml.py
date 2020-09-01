import logging
from collections import OrderedDict
from pathlib import Path
from typing import (
    Text,
    Any,
    List,
    Dict,
    Tuple,
    TYPE_CHECKING,
    Union,
    Iterator,
    Optional,
)

from rasa import data
from rasa.utils import validation
from ruamel.yaml import YAMLError, StringIO

import rasa.utils.io as io_utils
from rasa.constants import (
    DOCS_URL_TRAINING_DATA_NLU,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.nlu.training_data.formats.readerwriter import (
    TrainingDataReader,
    TrainingDataWriter,
)
from rasa.utils.common import raise_warning

if TYPE_CHECKING:
    from rasa.nlu.training_data import TrainingData, Message

logger = logging.getLogger(__name__)

KEY_NLU = "nlu"
KEY_RESPONSES = "responses"
KEY_INTENT = "intent"
KEY_INTENT_EXAMPLES = "examples"
KEY_INTENT_TEXT = "text"
KEY_SYNONYM = "synonym"
KEY_SYNONYM_EXAMPLES = "examples"
KEY_REGEX = "regex"
KEY_REGEX_EXAMPLES = "examples"
KEY_LOOKUP = "lookup"
KEY_LOOKUP_EXAMPLES = "examples"
KEY_METADATA = "metadata"

MULTILINE_TRAINING_EXAMPLE_LEADING_SYMBOL = "-"

NLU_SCHEMA_FILE = "nlu/schemas/nlu.yml"

STRIP_SYMBOLS = "\n\r "


class RasaYAMLReader(TrainingDataReader):
    """Reads YAML training data and creates a TrainingData object."""

    def __init__(self) -> None:
        super().__init__()
        self.training_examples: List[Message] = []
        self.entity_synonyms: Dict[Text, Text] = {}
        self.regex_features: List[Dict[Text, Text]] = []
        self.lookup_tables: List[Dict[Text, Any]] = []
        self.responses: Dict[Text, List[Dict[Text, Any]]] = {}

    @staticmethod
    def validate(string: Text) -> None:
        """Check if the string adheres to the NLU yaml data schema.

        If the string is not in the right format, an exception will be raised."""
        try:
            validation.validate_yaml_schema(string, NLU_SCHEMA_FILE)
        except validation.InvalidYamlFileError as e:
            raise ValueError from e

    def reads(self, string: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData in YAML format from a string.

        Args:
            string: String with YAML training data.
            **kwargs: Keyword arguments.

        Returns:
            New `TrainingData` object with parsed training data.
        """
        from rasa.nlu.training_data import TrainingData
        from rasa.validator import Validator

        self.validate(string)

        yaml_content = io_utils.read_yaml(string)

        if not Validator.validate_training_data_format_version(
            yaml_content, self.filename
        ):
            return TrainingData()

        for key, value in yaml_content.items():  # pytype: disable=attribute-error
            if key == KEY_NLU:
                self._parse_nlu(value)
            elif key == KEY_RESPONSES:
                self.responses = value

        return TrainingData(
            self.training_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
            self.responses,
        )

    def _parse_nlu(self, nlu_data: Optional[List[Dict[Text, Any]]]) -> None:

        if not nlu_data:
            return

        for nlu_item in nlu_data:
            if not isinstance(nlu_item, dict):
                raise_warning(
                    f"Unexpected block found in '{self.filename}':\n"
                    f"{nlu_item}\n"
                    f"Items under the '{KEY_NLU}' key must be YAML dictionaries. "
                    f"This block will be skipped.",
                    docs=DOCS_URL_TRAINING_DATA_NLU,
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
            else:
                raise_warning(
                    f"Issue found while processing '{self.filename}': "
                    f"Could not find supported key in the section:\n"
                    f"{nlu_item}\n"
                    f"Supported keys are: '{KEY_INTENT}', '{KEY_SYNONYM}', "
                    f"'{KEY_REGEX}', '{KEY_LOOKUP}'. "
                    f"This section will be skipped.",
                    docs=DOCS_URL_TRAINING_DATA_NLU,
                )

    def _parse_intent(self, intent_data: Dict[Text, Any]) -> None:
        from rasa.nlu.training_data import Message
        import rasa.nlu.training_data.entities_parser as entities_parser
        import rasa.nlu.training_data.synonyms_parser as synonyms_parser
        import rasa.nlu.constants as nlu_constants

        intent = intent_data.get(KEY_INTENT, "")
        if not intent:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The intent has an empty name. "
                f"Intents should have a name defined under the {KEY_INTENT} key. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = intent_data.get(KEY_INTENT_EXAMPLES, "")
        for example, entities in self._parse_training_examples(examples, intent):

            plain_text = entities_parser.replace_entities(example)

            synonyms_parser.add_synonyms_from_entities(
                plain_text, entities, self.entity_synonyms
            )

            message = Message.build(plain_text, intent)
            if entities:
                message.set(nlu_constants.ENTITIES, entities)
            self.training_examples.append(message)

    def _parse_training_examples(
        self, examples: Union[Text, List[Dict[Text, Any]]], intent: Text
    ) -> List[Tuple[Text, List[Dict[Text, Any]]]]:
        import rasa.nlu.training_data.entities_parser as entities_parser

        if isinstance(examples, list):
            example_strings = [
                # pytype: disable=attribute-error
                example.get(KEY_INTENT_TEXT, "").strip(STRIP_SYMBOLS)
                for example in examples
                if example
            ]
        # pytype: enable=attribute-error
        elif isinstance(examples, str):
            example_strings = self._parse_multiline_example(intent, examples)
        else:
            raise_warning(
                f"Unexpected block found in '{self.filename}' "
                f"while processing intent '{intent}':\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return []

        if not example_strings:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"Intent '{intent}' has no examples.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )

        results = []
        for example in example_strings:
            entities = entities_parser.find_entities_in_training_example(example)
            results.append((example, entities))

        return results

    def _parse_synonym(self, nlu_item: Dict[Text, Any]) -> None:
        import rasa.nlu.training_data.synonyms_parser as synonyms_parser

        synonym_name = nlu_item[KEY_SYNONYM]
        if not synonym_name:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The synonym has an empty name. "
                f"Synonyms should have a name defined under the {KEY_SYNONYM} key. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = nlu_item.get(KEY_SYNONYM_EXAMPLES, "")

        if not examples:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"{KEY_SYNONYM}: {synonym_name} doesn't have any examples. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        if not isinstance(examples, str):
            raise_warning(
                f"Unexpected block found in '{self.filename}':\n"
                f"{examples}\n"
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        for example in self._parse_multiline_example(synonym_name, examples):
            synonyms_parser.add_synonym(example, synonym_name, self.entity_synonyms)

    def _parse_regex(self, nlu_item: Dict[Text, Any]) -> None:
        regex_name = nlu_item[KEY_REGEX]
        if not regex_name:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The regex has an empty name."
                f"Regex should have a name defined under the '{KEY_REGEX}' key. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = nlu_item.get(KEY_REGEX_EXAMPLES, "")
        if not examples:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"'{KEY_REGEX}: {regex_name}' doesn't have any examples. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        if not isinstance(examples, str):
            raise_warning(
                f"Unexpected block found in '{self.filename}':\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        for example in self._parse_multiline_example(regex_name, examples):
            self.regex_features.append({"name": regex_name, "pattern": example})

    def _parse_lookup(self, nlu_item: Dict[Text, Any]):
        import rasa.nlu.training_data.lookup_tables_parser as lookup_tables_parser

        lookup_item_name = nlu_item[KEY_LOOKUP]
        if not lookup_item_name:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The lookup item has an empty name. "
                f"Lookup items should have a name defined under the '{KEY_LOOKUP}' "
                f"key. It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        examples = nlu_item.get(KEY_LOOKUP_EXAMPLES, "")
        if not examples:
            raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"'{KEY_LOOKUP}: {lookup_item_name}' doesn't have any examples. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        if not isinstance(examples, str):
            raise_warning(
                f"Unexpected block found in '{self.filename}':\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )
            return

        for example in self._parse_multiline_example(lookup_item_name, examples):
            lookup_tables_parser.add_item_to_lookup_tables(
                lookup_item_name, example, self.lookup_tables
            )

    def _parse_multiline_example(self, item: Text, examples: Text) -> Iterator[Text]:
        for example in examples.splitlines():
            if not example.startswith(MULTILINE_TRAINING_EXAMPLE_LEADING_SYMBOL):
                raise_warning(
                    f"Issue found while processing '{self.filename}': "
                    f"The item '{item}' contains an example that doesn't start with a "
                    f"'{MULTILINE_TRAINING_EXAMPLE_LEADING_SYMBOL}' symbol: "
                    f"{example}\n"
                    f"This training example will be skipped.",
                    docs=DOCS_URL_TRAINING_DATA_NLU,
                )
                continue
            yield example[1:].strip(STRIP_SYMBOLS)

    @staticmethod
    def is_yaml_nlu_file(filename: Text) -> bool:
        """Checks if the specified file possibly contains NLU training data in YAML.

        Args:
            filename: name of the file to check.

        Returns:
            `True` if the `filename` is possibly a valid YAML NLU file,
            `False` otherwise.
        """
        if not data.is_likely_yaml_file(filename):
            return False

        try:
            content = io_utils.read_yaml_file(filename)

            return any(key in content for key in {KEY_NLU, KEY_RESPONSES})
        except (YAMLError, Warning) as e:
            logger.error(
                f"Tried to check if '{filename}' is an NLU file, but failed to "
                f"read it. If this file contains NLU data, you should "
                f"investigate this error, otherwise it is probably best to "
                f"move the file to a different location. "
                f"Error: {e}"
            )
            return False


class RasaYAMLWriter(TrainingDataWriter):
    """Writes training data into a file in a YAML format."""

    def dumps(self, training_data: "TrainingData") -> Text:
        """Turns TrainingData into a string."""
        stream = StringIO()
        self.dump(stream, training_data)
        return stream.getvalue()

    def dump(
        self, target: Union[Text, Path, StringIO], training_data: "TrainingData"
    ) -> None:
        """Writes training data into a file in a YAML format.

        Args:
            target: Name of the target object to write the YAML to.
            training_data: TrainingData object.
        """
        from rasa.validator import KEY_TRAINING_DATA_FORMAT_VERSION
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString

        nlu_items = []
        nlu_items.extend(self.process_intents(training_data))
        nlu_items.extend(self.process_synonyms(training_data))
        nlu_items.extend(self.process_regexes(training_data))
        nlu_items.extend(self.process_lookup_tables(training_data))

        result = OrderedDict()
        result[KEY_TRAINING_DATA_FORMAT_VERSION] = DoubleQuotedScalarString(
            LATEST_TRAINING_DATA_FORMAT_VERSION
        )

        if nlu_items:
            result[KEY_NLU] = nlu_items

        if training_data.responses:
            result[KEY_RESPONSES] = training_data.responses

        io_utils.write_yaml(result, target, True)

    @classmethod
    def process_intents(cls, training_data: "TrainingData") -> List[OrderedDict]:
        training_data = cls.prepare_training_examples(training_data)
        return RasaYAMLWriter.process_training_examples_by_key(
            training_data,
            KEY_INTENT,
            KEY_INTENT_EXAMPLES,
            TrainingDataWriter.generate_message,
        )

    @classmethod
    def process_synonyms(cls, training_data: "TrainingData") -> List[OrderedDict]:
        inverted_synonyms = OrderedDict()
        for example, synonym in training_data.entity_synonyms.items():
            if not inverted_synonyms.get(synonym):
                inverted_synonyms[synonym] = []
            inverted_synonyms[synonym].append(example)

        return cls.process_training_examples_by_key(
            inverted_synonyms, KEY_SYNONYM, KEY_SYNONYM_EXAMPLES
        )

    @classmethod
    def process_regexes(cls, training_data: "TrainingData") -> List[OrderedDict]:
        inverted_regexes = OrderedDict()
        for regex in training_data.regex_features:
            if not inverted_regexes.get(regex["name"]):
                inverted_regexes[regex["name"]] = []
            inverted_regexes[regex["name"]].append(regex["pattern"])

        return cls.process_training_examples_by_key(
            inverted_regexes, KEY_REGEX, KEY_REGEX_EXAMPLES
        )

    @classmethod
    def process_lookup_tables(cls, training_data: "TrainingData") -> List[OrderedDict]:
        prepared_lookup_tables = OrderedDict()
        for lookup_table in training_data.lookup_tables:
            prepared_lookup_tables[lookup_table["name"]] = lookup_table["elements"]

        return cls.process_training_examples_by_key(
            prepared_lookup_tables, KEY_LOOKUP, KEY_LOOKUP_EXAMPLES
        )

    @staticmethod
    def process_training_examples_by_key(
        training_examples: Dict,
        key_name: Text,
        key_examples: Text,
        example_extraction_predicate=lambda x: x,
    ) -> List[OrderedDict]:
        from ruamel.yaml.scalarstring import LiteralScalarString

        result = []
        for entity_key, examples in training_examples.items():

            converted_examples = [
                TrainingDataWriter.generate_list_item(
                    example_extraction_predicate(example).strip(STRIP_SYMBOLS)
                )
                for example in examples
            ]

            next_item = OrderedDict()
            next_item[key_name] = entity_key
            next_item[key_examples] = LiteralScalarString("".join(converted_examples))
            result.append(next_item)

        return result
