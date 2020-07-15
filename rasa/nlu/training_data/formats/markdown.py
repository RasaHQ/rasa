import logging
import re
import typing
from collections import OrderedDict
from json import JSONDecodeError
from typing import Any, Text, Optional, Tuple, List, Dict, NamedTuple, Match

from rasa.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.core.constants import INTENT_MESSAGE_PREFIX

from rasa.nlu.training_data.formats.readerwriter import (
    TrainingDataReader,
    TrainingDataWriter,
)
from rasa.nlu.utils import build_entity
from rasa.utils.common import raise_warning
from rasa.nlu.constants import (
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_ROLE,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
)

GROUP_ENTITY_VALUE = "value"
GROUP_ENTITY_TYPE = "entity"
GROUP_ENTITY_DICT = "entity_dict"
GROUP_ENTITY_TEXT = "entity_text"
GROUP_COMPLETE_MATCH = 0

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import Message, TrainingData

INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
LOOKUP = "lookup"
available_sections = [INTENT, SYNONYM, REGEX, LOOKUP]

# regex for: `[entity_text]((entity_type(:entity_synonym)?)|{entity_dict})`
entity_regex = re.compile(
    r"\[(?P<entity_text>[^\]]+?)\](\((?P<entity>[^:)]+?)(?:\:(?P<value>[^)]+))?\)|\{(?P<entity_dict>[^}]+?)\})"
)
item_regex = re.compile(r"\s*[-*+]\s*(.+)")
comment_regex = re.compile(r"<!--[\s\S]*?--!*>", re.MULTILINE)
fname_regex = re.compile(r"\s*([^-*+]+)")

ESCAPE_DCT = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}

ESCAPE = re.compile(r"[\b\f\n\r\t]")


class EntityAttributes(NamedTuple):
    """Attributes of an entity defined in the markdown data."""

    type: Text
    value: Text
    text: Text
    group: Optional[Text]
    role: Optional[Text]


def encode_string(s: Text) -> Text:
    """Return a encoded python string."""

    def replace(match: Match) -> Text:
        return ESCAPE_DCT[match.group(GROUP_COMPLETE_MATCH)]

    return ESCAPE.sub(replace, s)


logger = logging.getLogger(__name__)


class MarkdownReader(TrainingDataReader):
    """Reads markdown training data and creates a TrainingData object."""

    def __init__(self) -> None:
        self.current_title = None
        self.current_section = None
        self.training_examples = []
        self.entity_synonyms = {}
        self.regex_features = []
        self.lookup_tables = []

        self._deprecated_synonym_format_was_used = False

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Read markdown string and create TrainingData object"""
        from rasa.nlu.training_data import TrainingData

        self.__init__()

        s = self._strip_comments(s)
        for line in s.splitlines():
            line = line.strip()
            header = self._find_section_header(line)
            if header:
                self._set_current_section(header[0], header[1])
            else:
                self._parse_item(line)
                self._load_files(line)

        if self._deprecated_synonym_format_was_used:
            raise_warning(
                "You are using the deprecated training data format to declare synonyms."
                " Please use the following format: \n"
                '[<entity-text>]{"entity": "<entity-type>", "value": '
                '"<entity-synonym>"}.'
                "\nYou can use the following command to update your training data file:"
                "\nsed -i -E 's/\\[([^)]+)\\]\\(([^)]+):([^)]+)\\)/[\\1]{"
                '"entity": "\\2", "value": "\\3"}/g\' nlu.md',
                category=FutureWarning,
                docs=DOCS_URL_TRAINING_DATA_NLU,
            )

        return TrainingData(
            self.training_examples,
            self.entity_synonyms,
            self.regex_features,
            self.lookup_tables,
        )

    @staticmethod
    def _strip_comments(text: Text) -> Text:
        """ Removes comments defined by `comment_regex` from `text`. """
        return re.sub(comment_regex, "", text)

    @staticmethod
    def _find_section_header(line: Text) -> Optional[Tuple[Text, Text]]:
        """Checks if the current line contains a section header
        and returns the section and the title."""
        match = re.search(r"##\s*(.+?):(.+)", line)
        if match is not None:
            return match.group(1), match.group(2)

        return None

    def _load_files(self, line: Text) -> None:
        """Checks line to see if filename was supplied.  If so, inserts the
        filename into the lookup table slot for processing from the regex
        featurizer."""
        if self.current_section == LOOKUP:
            match = re.match(fname_regex, line)
            if match:
                fname = line.strip()
                self.lookup_tables.append(
                    {"name": self.current_title, "elements": str(fname)}
                )

    def _parse_item(self, line: Text) -> None:
        """Parses an md list item line based on the current section type."""
        match = re.match(item_regex, line)
        if match:
            item = match.group(1)
            if self.current_section == INTENT:
                parsed = self.parse_training_example(item)
                self.training_examples.append(parsed)
            elif self.current_section == SYNONYM:
                self._add_synonym(item, self.current_title)
            elif self.current_section == REGEX:
                self.regex_features.append(
                    {"name": self.current_title, "pattern": item}
                )
            elif self.current_section == LOOKUP:
                self._add_item_to_lookup(item)

    def _add_item_to_lookup(self, item: Text) -> None:
        """Takes a list of lookup table dictionaries.  Finds the one associated
        with the current lookup, then adds the item to the list."""
        matches = [
            table for table in self.lookup_tables if table["name"] == self.current_title
        ]
        if not matches:
            self.lookup_tables.append({"name": self.current_title, "elements": [item]})
        else:
            elements = matches[0]["elements"]
            elements.append(item)

    @staticmethod
    def _get_validated_dict(json_str: Text) -> Dict[Text, Text]:
        """Converts the provided json_str to a valid dict containing the entity
        attributes.

        Users can specify entity roles, synonyms, groups for an entity in a dict, e.g.
        [LA]{"entity": "city", "role": "to", "value": "Los Angeles"}

        Args:
            json_str: the entity dict as string without "{}"

        Raises:
            ValidationError if validation of entity dict fails.
            JSONDecodeError if provided entity dict is not valid json.

        Returns:
            a proper python dict
        """
        import json
        import rasa.utils.validation as validation_utils
        import rasa.nlu.schemas.data_schema as schema

        # add {} as they are not part of the regex
        try:
            data = json.loads(f"{{{json_str}}}")
        except JSONDecodeError as e:
            raise_warning(
                f"Incorrect training data format ('{{{json_str}}}'), make sure your "
                f"data is valid. For more information about the format visit "
                f"{DOCS_URL_TRAINING_DATA_NLU}."
            )
            raise e

        validation_utils.validate_training_data(data, schema.entity_dict_schema())

        return data

    def _find_entities_in_training_example(self, example: Text) -> List[Dict]:
        """Extracts entities from a markdown intent example.

        Args:
            example: markdown intent example

        Returns: list of extracted entities
        """
        entities = []
        offset = 0

        for match in re.finditer(entity_regex, example):
            entity_attributes = self._extract_entity_attributes(match)

            start_index = match.start() - offset
            end_index = start_index + len(entity_attributes.text)
            offset += len(match.group(0)) - len(entity_attributes.text)

            entity = build_entity(
                start_index,
                end_index,
                entity_attributes.value,
                entity_attributes.type,
                entity_attributes.role,
                entity_attributes.group,
            )
            entities.append(entity)

        return entities

    def _extract_entity_attributes(self, match: Match) -> EntityAttributes:
        """Extract the entity attributes, i.e. type, value, etc., from the
        regex match."""
        entity_text = match.groupdict()[GROUP_ENTITY_TEXT]

        if match.groupdict()[GROUP_ENTITY_DICT]:
            return self._extract_entity_attributes_from_dict(entity_text, match)

        entity_type = match.groupdict()[GROUP_ENTITY_TYPE]

        if match.groupdict()[GROUP_ENTITY_VALUE]:
            entity_value = match.groupdict()[GROUP_ENTITY_VALUE]
            self._deprecated_synonym_format_was_used = True
        else:
            entity_value = entity_text

        return EntityAttributes(entity_type, entity_value, entity_text, None, None)

    def _extract_entity_attributes_from_dict(
        self, entity_text: Text, match: Match
    ) -> EntityAttributes:
        """Extract the entity attributes from the dict format."""
        entity_dict_str = match.groupdict()[GROUP_ENTITY_DICT]
        entity_dict = self._get_validated_dict(entity_dict_str)
        return EntityAttributes(
            entity_dict.get(ENTITY_ATTRIBUTE_TYPE),
            entity_dict.get(ENTITY_ATTRIBUTE_VALUE, entity_text),
            entity_text,
            entity_dict.get(ENTITY_ATTRIBUTE_GROUP),
            entity_dict.get(ENTITY_ATTRIBUTE_ROLE),
        )

    def _add_synonym(self, text: Text, value: Text) -> None:
        from rasa.nlu.training_data.util import check_duplicate_synonym

        check_duplicate_synonym(self.entity_synonyms, text, value, "reading markdown")
        self.entity_synonyms[text] = value

    def _add_synonyms(self, plain_text: Text, entities: List[Dict]) -> None:
        """Adds synonyms found in intent examples"""
        for e in entities:
            e_text = plain_text[e[ENTITY_ATTRIBUTE_START] : e[ENTITY_ATTRIBUTE_END]]
            if e_text != e[ENTITY_ATTRIBUTE_VALUE]:
                self._add_synonym(e_text, e[ENTITY_ATTRIBUTE_VALUE])

    def parse_training_example(self, example: Text) -> "Message":
        """Extract entities and synonyms, and convert to plain text."""
        from rasa.nlu.training_data import Message

        entities = self._find_entities_in_training_example(example)
        plain_text = re.sub(
            entity_regex, lambda m: m.groupdict()[GROUP_ENTITY_TEXT], example
        )
        self._add_synonyms(plain_text, entities)

        message = Message.build(plain_text, self.current_title)

        if len(entities) > 0:
            message.set("entities", entities)
        return message

    def _set_current_section(self, section: Text, title: Text) -> None:
        """Update parsing mode."""
        if section not in available_sections:
            raise ValueError(
                "Found markdown section '{}' which is not "
                "in the allowed sections '{}'."
                "".format(section, "', '".join(available_sections))
            )

        self.current_section = section
        self.current_title = title


class MarkdownWriter(TrainingDataWriter):
    def dumps(self, training_data: "TrainingData") -> Text:
        """Transforms a TrainingData object into a markdown string."""

        md = ""
        md += self._generate_training_examples_md(training_data)
        md += self._generate_synonyms_md(training_data)
        md += self._generate_regex_features_md(training_data)
        md += self._generate_lookup_tables_md(training_data)

        return md

    def _generate_training_examples_md(self, training_data: "TrainingData") -> Text:
        """Generates markdown training examples."""

        import rasa.nlu.training_data.util as rasa_nlu_training_data_utils

        training_examples = OrderedDict()

        # Sort by intent while keeping basic intent order
        for example in [e.as_dict_nlu() for e in training_data.training_examples]:
            rasa_nlu_training_data_utils.remove_untrainable_entities_from(example)
            intent = example[INTENT]
            training_examples.setdefault(intent, [])
            training_examples[intent].append(example)

        # Don't prepend newline for first line
        prepend_newline = False
        lines = []

        for intent, examples in training_examples.items():
            section_header = self._generate_section_header_md(
                INTENT, intent, prepend_newline=prepend_newline
            )
            lines.append(section_header)
            prepend_newline = True

            lines += [
                self._generate_item_md(self.generate_message_md(example))
                for example in examples
            ]

        return "".join(lines)

    def _generate_synonyms_md(self, training_data: "TrainingData") -> Text:
        """Generates markdown for entity synomyms."""

        entity_synonyms = sorted(
            training_data.entity_synonyms.items(), key=lambda x: x[1]
        )
        md = ""
        for i, synonym in enumerate(entity_synonyms):
            if i == 0 or entity_synonyms[i - 1][1] != synonym[1]:
                md += self._generate_section_header_md(SYNONYM, synonym[1])

            md += self._generate_item_md(synonym[0])

        return md

    def _generate_regex_features_md(self, training_data: "TrainingData") -> Text:
        """Generates markdown for regex features."""

        md = ""
        # regex features are already sorted
        regex_features = training_data.regex_features
        for i, regex_feature in enumerate(regex_features):
            if i == 0 or regex_features[i - 1]["name"] != regex_feature["name"]:
                md += self._generate_section_header_md(REGEX, regex_feature["name"])

            md += self._generate_item_md(regex_feature["pattern"])

        return md

    def _generate_lookup_tables_md(self, training_data: "TrainingData") -> Text:
        """Generates markdown for regex features."""

        md = ""
        # regex features are already sorted
        lookup_tables = training_data.lookup_tables
        for lookup_table in lookup_tables:
            md += self._generate_section_header_md(LOOKUP, lookup_table["name"])
            elements = lookup_table["elements"]
            if isinstance(elements, list):
                for e in elements:
                    md += self._generate_item_md(e)
            else:
                md += self._generate_fname_md(elements)
        return md

    @staticmethod
    def _generate_section_header_md(
        section_type: Text, title: Text, prepend_newline: bool = True
    ) -> Text:
        """Generates markdown section header."""

        prefix = "\n" if prepend_newline else ""
        title = encode_string(title)

        return f"{prefix}## {section_type}:{title}\n"

    @staticmethod
    def _generate_item_md(text: Text) -> Text:
        """Generates markdown for a list item."""

        return f"- {encode_string(text)}\n"

    @staticmethod
    def _generate_fname_md(text: Text) -> Text:
        """Generates markdown for a lookup table file path."""

        return f"  {encode_string(text)}\n"

    @staticmethod
    def generate_message_md(message: Dict[Text, Any]) -> Text:
        """Generates markdown for a message object."""

        md = ""
        text = message.get("text", "")

        pos = 0

        # If a message was prefixed with `INTENT_MESSAGE_PREFIX` (this can only happen
        # in end-to-end stories) then potential entities were provided in the json
        # format (e.g. `/greet{"name": "Rasa"}) and we don't have to add the NLU
        # entity annotation
        if not text.startswith(INTENT_MESSAGE_PREFIX):
            entities = sorted(message.get("entities", []), key=lambda k: k["start"])

            for entity in entities:
                md += text[pos : entity["start"]]
                md += MarkdownWriter.generate_entity_md(text, entity)
                pos = entity["end"]

        md += text[pos:]

        return md

    @staticmethod
    def generate_entity_md(text: Text, entity: Dict[Text, Any]) -> Text:
        """Generates markdown for an entity object."""
        import json

        entity_text = text[
            entity[ENTITY_ATTRIBUTE_START] : entity[ENTITY_ATTRIBUTE_END]
        ]
        entity_type = entity.get(ENTITY_ATTRIBUTE_TYPE)
        entity_value = entity.get(ENTITY_ATTRIBUTE_VALUE)
        entity_role = entity.get(ENTITY_ATTRIBUTE_ROLE)
        entity_group = entity.get(ENTITY_ATTRIBUTE_GROUP)

        if entity_value and entity_value == entity_text:
            entity_value = None

        use_short_syntax = (
            entity_value is None and entity_role is None and entity_group is None
        )

        if use_short_syntax:
            return f"[{entity_text}]({entity_type})"

        entity_dict = {
            ENTITY_ATTRIBUTE_TYPE: entity_type,
            ENTITY_ATTRIBUTE_ROLE: entity_role,
            ENTITY_ATTRIBUTE_GROUP: entity_group,
            ENTITY_ATTRIBUTE_VALUE: entity_value,
        }
        entity_dict = {k: v for k, v in entity_dict.items() if v is not None}

        return f"[{entity_text}]{json.dumps(entity_dict)}"
