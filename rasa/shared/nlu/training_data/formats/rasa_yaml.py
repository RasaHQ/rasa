import logging
from collections import OrderedDict
from pathlib import Path
from typing import Text, Any, List, Dict, Tuple, Union, Iterator, Optional, Callable

import rasa.shared.data
from rasa.shared.core.domain import Domain
from rasa.shared.exceptions import YamlException
from ruamel.yaml import StringIO
from ruamel.yaml.scalarstring import LiteralScalarString

from rasa.shared.constants import (
    DOCS_URL_TRAINING_DATA,
    LATEST_TRAINING_DATA_FORMAT_VERSION,
)
from rasa.shared.nlu.constants import METADATA_INTENT, METADATA_EXAMPLE
from rasa.shared.nlu.training_data.formats.readerwriter import (
    TrainingDataReader,
    TrainingDataWriter,
)
import rasa.shared.utils.io
import rasa.shared.nlu.training_data.util
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.yaml import (
    validate_raw_yaml_using_schema_file_with_responses,
    validate_training_data_format_version,
    read_yaml,
    is_key_in_yaml,
    write_yaml,
)

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

NLU_SCHEMA_FILE = "shared/nlu/training_data/schemas/nlu.yml"

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

    def validate(self, string: Text) -> None:
        """Check if the string adheres to the NLU yaml data schema.

        If the string is not in the right format, an exception will be raised.
        """
        try:
            validate_raw_yaml_using_schema_file_with_responses(string, NLU_SCHEMA_FILE)
        except YamlException as e:
            e.filename = self.filename
            raise e

    def reads(  # type: ignore[override]
        self, string: Text, **kwargs: Any
    ) -> "TrainingData":
        """Reads TrainingData in YAML format from a string.

        Args:
            string: String with YAML training data.
            **kwargs: Keyword arguments.

        Returns:
            New `TrainingData` object with parsed training data.
        """
        self.validate(string)

        yaml_content = read_yaml(string)

        if not validate_training_data_format_version(yaml_content, self.filename):
            return TrainingData()

        for key, value in yaml_content.items():
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
                rasa.shared.utils.io.raise_warning(
                    f"Unexpected block found in '{self.filename}':\n"
                    f"{nlu_item}\n"
                    f"Items under the '{KEY_NLU}' key must be YAML dictionaries. "
                    f"This block will be skipped.",
                    docs=DOCS_URL_TRAINING_DATA,
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
                rasa.shared.utils.io.raise_warning(
                    f"Issue found while processing '{self.filename}': "
                    f"Could not find supported key in the section:\n"
                    f"{nlu_item}\n"
                    f"Supported keys are: '{KEY_INTENT}', '{KEY_SYNONYM}', "
                    f"'{KEY_REGEX}', '{KEY_LOOKUP}'. "
                    f"This section will be skipped.",
                    docs=DOCS_URL_TRAINING_DATA,
                )

    def _parse_intent(self, intent_data: Dict[Text, Any]) -> None:
        import rasa.shared.nlu.training_data.entities_parser as entities_parser
        import rasa.shared.nlu.training_data.synonyms_parser as synonyms_parser

        intent = intent_data.get(KEY_INTENT, "")
        if not intent:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The intent has an empty name. "
                f"Intents should have a name defined under the {KEY_INTENT} key. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        examples = intent_data.get(KEY_INTENT_EXAMPLES, "")
        intent_metadata = intent_data.get(KEY_METADATA)
        for example, entities, metadata in self._parse_training_examples(
            examples, intent
        ):
            plain_text = entities_parser.replace_entities(example)

            synonyms_parser.add_synonyms_from_entities(
                plain_text, entities, self.entity_synonyms
            )

            self.training_examples.append(
                Message.build(plain_text, intent, entities, intent_metadata, metadata)
            )

    def _parse_training_examples(
        self, examples: Union[Text, List[Dict[Text, Any]]], intent: Text
    ) -> List[Tuple[Text, List[Dict[Text, Any]], Optional[Any]]]:
        import rasa.shared.nlu.training_data.entities_parser as entities_parser

        if isinstance(examples, list):
            example_tuples = [
                (
                    example.get(KEY_INTENT_TEXT, "").strip(STRIP_SYMBOLS),
                    example.get(KEY_METADATA),
                )
                for example in examples
                if example
            ]
        elif isinstance(examples, str):
            example_tuples = [
                (example, None)
                for example in self._parse_multiline_example(intent, examples)
            ]
        else:
            rasa.shared.utils.io.raise_warning(
                f"Unexpected block found in '{self.filename}' "
                f"while processing intent '{intent}':\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return []

        if not example_tuples:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"Intent '{intent}' has no examples.",
                docs=DOCS_URL_TRAINING_DATA,
            )

        results = []
        for example, metadata in example_tuples:
            entities = entities_parser.find_entities_in_training_example(example)
            results.append((example, entities, metadata))

        return results

    def _parse_synonym(self, nlu_item: Dict[Text, Any]) -> None:
        import rasa.shared.nlu.training_data.synonyms_parser as synonyms_parser

        synonym_name = nlu_item[KEY_SYNONYM]
        if not synonym_name:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The synonym has an empty name. "
                f"Synonyms should have a name defined under the {KEY_SYNONYM} key. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        examples = nlu_item.get(KEY_SYNONYM_EXAMPLES, "")

        if not examples:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"{KEY_SYNONYM}: {synonym_name} doesn't have any examples. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        if not isinstance(examples, str):
            rasa.shared.utils.io.raise_warning(
                f"Unexpected block found in '{self.filename}':\n"
                f"{examples}\n"
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        for example in self._parse_multiline_example(synonym_name, examples):
            synonyms_parser.add_synonym(example, synonym_name, self.entity_synonyms)

    def _parse_regex(self, nlu_item: Dict[Text, Any]) -> None:
        regex_name = nlu_item[KEY_REGEX]
        if not regex_name:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The regex has an empty name."
                f"Regex should have a name defined under the '{KEY_REGEX}' key. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        examples = nlu_item.get(KEY_REGEX_EXAMPLES, "")
        if not examples:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"'{KEY_REGEX}: {regex_name}' doesn't have any examples. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        if not isinstance(examples, str):
            rasa.shared.utils.io.raise_warning(
                f"Unexpected block found in '{self.filename}':\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        for example in self._parse_multiline_example(regex_name, examples):
            self.regex_features.append({"name": regex_name, "pattern": example})

    def _parse_lookup(self, nlu_item: Dict[Text, Any]) -> None:
        import rasa.shared.nlu.training_data.lookup_tables_parser as lookup_tables_parser  # noqa: E501

        lookup_item_name = nlu_item[KEY_LOOKUP]
        if not lookup_item_name:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"The lookup item has an empty name. "
                f"Lookup items should have a name defined under the '{KEY_LOOKUP}' "
                f"key. It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        examples = nlu_item.get(KEY_LOOKUP_EXAMPLES, "")
        if not examples:
            rasa.shared.utils.io.raise_warning(
                f"Issue found while processing '{self.filename}': "
                f"'{KEY_LOOKUP}: {lookup_item_name}' doesn't have any examples. "
                f"It will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        if not isinstance(examples, str):
            rasa.shared.utils.io.raise_warning(
                f"Unexpected block found in '{self.filename}':\n"
                f"{examples}\n"
                f"This block will be skipped.",
                docs=DOCS_URL_TRAINING_DATA,
            )
            return

        for example in self._parse_multiline_example(lookup_item_name, examples):
            lookup_tables_parser.add_item_to_lookup_tables(
                lookup_item_name, example, self.lookup_tables
            )

    def _parse_multiline_example(self, item: Text, examples: Text) -> Iterator[Text]:
        for example in examples.splitlines():
            if not example.startswith(MULTILINE_TRAINING_EXAMPLE_LEADING_SYMBOL):
                rasa.shared.utils.io.raise_warning(
                    f"Issue found while processing '{self.filename}': "
                    f"The item '{item}' contains an example that doesn't start with a "
                    f"'{MULTILINE_TRAINING_EXAMPLE_LEADING_SYMBOL}' symbol: "
                    f"{example}\n"
                    f"This training example will be skipped.",
                    docs=DOCS_URL_TRAINING_DATA,
                )
                continue
            yield example[1:].strip(STRIP_SYMBOLS)

    @staticmethod
    def is_yaml_nlu_file(filename: Union[Text, Path]) -> bool:
        """Checks if the specified file possibly contains NLU training data in YAML.

        Args:
            filename: name of the file to check.

        Returns:
            `True` if the `filename` is possibly a valid YAML NLU file,
            `False` otherwise.

        Raises:
            YamlException: if the file seems to be a YAML file (extension) but
                can not be read / parsed.
        """
        if not rasa.shared.data.is_likely_yaml_file(filename):
            return False

        return is_key_in_yaml(filename, KEY_NLU, KEY_RESPONSES)


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
        result = self.training_data_to_dict(training_data)

        if result:
            write_yaml(result, target, True)

    @classmethod
    def training_data_to_dict(
        cls, training_data: "TrainingData"
    ) -> Optional[OrderedDict]:
        """Represents NLU training data to a dict/list structure ready to be
        serialized as YAML.

        Args:
            training_data: `TrainingData` to convert.

        Returns:
            `OrderedDict` containing all training data.
        """
        from rasa.shared.utils.yaml import KEY_TRAINING_DATA_FORMAT_VERSION
        from ruamel.yaml.scalarstring import DoubleQuotedScalarString

        nlu_items = []
        nlu_items.extend(cls.process_intents(training_data))
        nlu_items.extend(cls.process_synonyms(training_data))
        nlu_items.extend(cls.process_regexes(training_data))
        nlu_items.extend(cls.process_lookup_tables(training_data))

        if not any([nlu_items, training_data.responses]):
            return None

        result: OrderedDict[Text, Any] = OrderedDict()
        result[KEY_TRAINING_DATA_FORMAT_VERSION] = DoubleQuotedScalarString(
            LATEST_TRAINING_DATA_FORMAT_VERSION
        )

        if nlu_items:
            result[KEY_NLU] = nlu_items

        if training_data.responses:
            result[KEY_RESPONSES] = Domain.get_responses_with_multilines(
                training_data.responses
            )

        return result

    @classmethod
    def process_intents(cls, training_data: "TrainingData") -> List[OrderedDict]:
        """Serializes the intents."""
        return RasaYAMLWriter.process_training_examples_by_key(
            cls.prepare_training_examples(training_data),
            KEY_INTENT,
            KEY_INTENT_EXAMPLES,
            TrainingDataWriter.generate_message,
        )

    @classmethod
    def process_synonyms(cls, training_data: "TrainingData") -> List[OrderedDict]:
        """Serializes the synonyms."""
        inverted_synonyms: Dict[Text, List[Dict]] = OrderedDict()
        for example, synonym in training_data.entity_synonyms.items():
            if not inverted_synonyms.get(synonym):
                inverted_synonyms[synonym] = []
            inverted_synonyms[synonym].append(example)

        return cls.process_training_examples_by_key(
            inverted_synonyms,
            KEY_SYNONYM,
            KEY_SYNONYM_EXAMPLES,
            example_extraction_predicate=lambda x: str(x),
        )

    @classmethod
    def process_regexes(cls, training_data: "TrainingData") -> List[OrderedDict]:
        """Serializes the regexes."""
        inverted_regexes: Dict[Text, List[Text]] = OrderedDict()
        for regex in training_data.regex_features:
            if not inverted_regexes.get(regex["name"]):
                inverted_regexes[regex["name"]] = []
            inverted_regexes[regex["name"]].append(regex["pattern"])

        return cls.process_training_examples_by_key(
            inverted_regexes,
            KEY_REGEX,
            KEY_REGEX_EXAMPLES,
            example_extraction_predicate=lambda x: str(x),
        )

    @classmethod
    def process_lookup_tables(cls, training_data: "TrainingData") -> List[OrderedDict]:
        """Serializes the look up tables.

        Args:
            training_data: The training data object with potential look up tables.

        Returns:
            The serialized lookup tables.
        """
        prepared_lookup_tables: Dict[Text, List[Text]] = OrderedDict()
        for lookup_table in training_data.lookup_tables:
            # this is a lookup table filename
            if isinstance(lookup_table["elements"], str):
                continue
            prepared_lookup_tables[lookup_table["name"]] = lookup_table["elements"]

        return cls.process_training_examples_by_key(
            prepared_lookup_tables,
            KEY_LOOKUP,
            KEY_LOOKUP_EXAMPLES,
            example_extraction_predicate=lambda x: str(x),
        )

    @staticmethod
    def process_training_examples_by_key(
        training_examples: Dict[Text, List[Union[Dict, Text]]],
        key_name: Text,
        key_examples: Text,
        example_extraction_predicate: Callable[[Dict[Text, Any]], Text],
    ) -> List[OrderedDict]:
        """Prepares training examples  to be written to YAML.

        This can be any NLU training data (intent examples, lookup tables, etc.)

        Args:
            training_examples: Multiple training examples. Mappings in case additional
                values were specified for an example (e.g. metadata) or just the plain
                value.
            key_name: The top level key which the examples belong to (e.g. `intents`)
            key_examples: The sub key which the examples should be added to
                (e.g. `examples`).
            example_extraction_predicate: Function to extract example value (e.g. the
                the text for an intent example)

        Returns:
            NLU training data examples prepared for writing to YAML.
        """
        intents = []

        for intent_name, examples in training_examples.items():
            converted, intent_metadata = RasaYAMLWriter._convert_training_examples(
                examples, example_extraction_predicate
            )

            intent: OrderedDict[Text, Any] = OrderedDict()
            intent[key_name] = intent_name
            if intent_metadata:
                intent[KEY_METADATA] = intent_metadata

            examples_have_metadata = any(KEY_METADATA in ex for ex in converted)
            example_texts_have_escape_chars = any(
                rasa.shared.nlu.training_data.util.has_string_escape_chars(
                    ex.get(KEY_INTENT_TEXT, "")
                )
                for ex in converted
            )

            if examples_have_metadata or example_texts_have_escape_chars:
                intent[key_examples] = (
                    RasaYAMLWriter._render_training_examples_as_objects(converted)
                )
            else:
                intent[key_examples] = RasaYAMLWriter._render_training_examples_as_text(
                    converted
                )

            intents.append(intent)

        return intents

    @staticmethod
    def _convert_training_examples(
        training_examples: List[Union[Dict, List[Text]]],
        example_extraction_predicate: Callable[[Dict[Text, Any]], Text],
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """Returns converted training examples and potential intent metadata."""
        converted_examples = []
        intent_metadata = None

        for example in training_examples:
            converted = {
                KEY_INTENT_TEXT: example_extraction_predicate(example).strip(
                    STRIP_SYMBOLS
                )
            }

            if isinstance(example, dict) and KEY_METADATA in example:
                metadata = example[KEY_METADATA]

                if METADATA_EXAMPLE in metadata:
                    converted[KEY_METADATA] = metadata[METADATA_EXAMPLE]

                if intent_metadata is None and METADATA_INTENT in metadata:
                    intent_metadata = metadata[METADATA_INTENT]

            converted_examples.append(converted)

        return converted_examples, intent_metadata

    @staticmethod
    def _render_training_examples_as_objects(examples: List[Dict]) -> List[Dict]:
        """Renders training examples as objects.

        The `text` item is rendered as a literal scalar string.

        Given the input of a single example:
            {'text': 'how much CO2 will that use?'}
        Its return value is a dictionary that will be rendered in YAML as:
        ```
            text: |
              how much CO2 will that use?
        ```
        """

        def render(example: Dict) -> Dict:
            text = example[KEY_INTENT_TEXT]
            example[KEY_INTENT_TEXT] = LiteralScalarString(text + "\n")
            return example

        return [render(ex) for ex in examples]

    @staticmethod
    def _render_training_examples_as_text(examples: List[Dict]) -> LiteralScalarString:
        def render(example: Dict) -> Text:
            return TrainingDataWriter.generate_list_item(example[KEY_INTENT_TEXT])

        return LiteralScalarString("".join([render(example) for example in examples]))
