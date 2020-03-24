import logging
import re
import typing
from collections import OrderedDict
from typing import Any, Text, Optional, Tuple, List, Dict, NamedTuple, Match

from rasa.constants import DOCS_URL_TRAINING_DATA_NLU
from rasa.core.constants import INTENT_MESSAGE_PREFIX

from rasa.nlu.training_data.formats.readerwriter import (
    TrainingDataReader,
    TrainingDataWriter,
)
from rasa.nlu.utils import build_entity
from rasa.nlu.constants import INTENT
from rasa.utils.common import raise_warning

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import Message, TrainingData

INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
LOOKUP = "lookup"
available_sections = [INTENT, SYNONYM, REGEX, LOOKUP]

# regex for: `[entity_text]((entity_type(:entity_synonym)?)|{entity_dict})`
entity_regex = re.compile(
    r"\[(?P<entity_text>[^\]]+)\](\((?P<entity>[^:)]*?)(?:\:(?P<value>[^)]+))?\)|\{(?P<entity_dict>[^}]*?)\})"
)
item_regex = re.compile(r"\s*[-*+]\s*(.+)")
comment_regex = re.compile(r"<!--[\s\S]*?--!*>", re.MULTILINE)
fname_regex = re.compile(r"\s*([^-*+]+)")

ESCAPE_DCT = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}

ESCAPE = re.compile(r"[\b\f\n\r\t]")


class EntityValues(NamedTuple):
    type: Text
    value: Text
    text: Text
    group: Optional[Text]
    role: Optional[Text]


def encode_string(s: Text) -> Text:
    """Return a encoded python string."""

    def replace(match):
        return ESCAPE_DCT[match.group(0)]

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

    def _find_section_header(self, line: Text) -> Optional[Tuple[Text, Text]]:
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
        matches = [l for l in self.lookup_tables if l["name"] == self.current_title]
        if not matches:
            self.lookup_tables.append({"name": self.current_title, "elements": [item]})
        else:
            elements = matches[0]["elements"]
            elements.append(item)

    @staticmethod
    def _entity_dict_schema() -> Dict[Text, Text]:
        return {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "role": {"type": "string"},
                "group": {"type": "string"},
                "synonym": {"type": "string"},
            },
            "required": ["entity"],
        }

    def _validate_entity_dict(self, json_str: Text) -> Dict[Text, Text]:
        """
        Validates the entity dict data.

        Users can specify entity roles, synonyms, groups for an entity in a dict, e.g.
        [LA]{"entity": "city", "role": "to", "synonym": "Los Angeles"}

        Args:
            json_str: the entity dict as string without "{}"

        Returns: a proper python dict
        """
        from jsonschema import validate
        from jsonschema import ValidationError
        import json

        # add {} as they are not part of the regex
        data = json.loads(f"{{{json_str}}}")

        try:
            validate(data, self._entity_dict_schema())
        except ValidationError as e:
            e.message += (
                f". Invalid entity format is used in the training data. "
                f"For more information about the format visit "
                f"{DOCS_URL_TRAINING_DATA_NLU}."
            )
            raise e

        return data

    def _find_entities_in_training_example(self, example: Text) -> List[Dict]:
        """Extracts entities from a markdown intent example."""
        entities = []
        offset = 0

        for match in re.finditer(entity_regex, example):
            entity_values = self._extract_entity_values(match)

            start_index = match.start() - offset
            end_index = start_index + len(entity_values.text)
            offset += len(match.group(0)) - len(entity_values.text)

            entity = build_entity(
                start_index,
                end_index,
                entity_values.value,
                entity_values.type,
                entity_values.role,
                entity_values.group,
            )
            entities.append(entity)

        return entities

    def _extract_entity_values(self, match: Match) -> EntityValues:
        """Extract the entity values, i.e. type, value, etc, from the regex match."""
        entity_text = match.groupdict()["entity_text"]

        if match.groupdict()["entity_dict"]:
            entity_dict_str = match.groupdict()["entity_dict"]
            entity_dict = self._validate_entity_dict(entity_dict_str)

            entity_type = entity_dict["entity"]
            entity_value = (
                entity_dict["synonym"] if "synonym" in entity_dict else entity_text
            )
            entity_role = entity_dict["role"] if "role" in entity_dict else None
            entity_group = entity_dict["group"] if "group" in entity_dict else None
        else:
            entity_type = match.groupdict()["entity"]
            entity_role = None
            entity_group = None

            if match.groupdict()["value"]:
                entity_value = match.groupdict()["value"]
                raise_warning(
                    "You are using the deprecated training data format to "
                    "declare synonyms. Please use the following format: "
                    "[<entity-text>]{'entity': <entity-type>, 'synonym': "
                    "<entity-synonym>}.",
                    category=DeprecationWarning,
                    docs=DOCS_URL_TRAINING_DATA_NLU,
                )
            else:
                entity_value = entity_text

        return EntityValues(
            entity_type, entity_value, entity_text, entity_group, entity_role
        )

    def _add_synonym(self, text: Text, value: Text) -> None:
        from rasa.nlu.training_data.util import check_duplicate_synonym

        check_duplicate_synonym(self.entity_synonyms, text, value, "reading markdown")
        self.entity_synonyms[text] = value

    def _add_synonyms(self, plain_text: Text, entities: List[Dict]) -> None:
        """Adds synonyms found in intent examples"""
        for e in entities:
            e_text = plain_text[e["start"] : e["end"]]
            if e_text != e["value"]:
                self._add_synonym(e_text, e["value"])

    def parse_training_example(self, example: Text) -> "Message":
        """Extract entities and synonyms, and convert to plain text."""
        from rasa.nlu.training_data import Message

        entities = self._find_entities_in_training_example(example)
        plain_text = re.sub(
            entity_regex, lambda m: m.groupdict()["entity_text"], example
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
        for i, lookup_table in enumerate(lookup_tables):
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
    def generate_entity_md(text: Text, entity: Dict) -> Text:
        """Generates markdown for an entity object."""

        entity_text = text[entity["start"] : entity["end"]]
        entity_type = entity["entity"]
        entity_synonym = (
            entity["value"]
            if "value" in entity and entity["value"] != entity_text
            else None
        )
        entity_role = entity["role"] if "role" in entity else None
        entity_group = entity["group"] if "group" in entity else None

        if entity_synonym is None and entity_role is None and entity_group is None:
            return f"[{entity_text}]({entity_type})"

        entity_dict_str = f'"entity": "{entity_type}"'
        if entity_role is not None:
            entity_dict_str += f', "role": "{entity_role}"'
        if entity_group is not None:
            entity_dict_str += f', "group": "{entity_group}"'
        if entity_synonym is not None:
            entity_dict_str += f', "synonym": "{entity_synonym}"'

        return f"[{entity_text}]{{{entity_dict_str}}}"
