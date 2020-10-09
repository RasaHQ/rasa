import logging
import re
from collections import OrderedDict
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Text, Optional, Tuple, Dict, Union

from rasa.shared.constants import LEGACY_DOCS_BASE_URL
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.formats.readerwriter import (
    TrainingDataReader,
    TrainingDataWriter,
)
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.util import encode_string, decode_string
from rasa.shared.nlu.training_data.training_data import TrainingData

GROUP_ENTITY_VALUE = "value"
GROUP_ENTITY_TYPE = "entity"
GROUP_ENTITY_DICT = "entity_dict"
GROUP_ENTITY_TEXT = "entity_text"


INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
LOOKUP = "lookup"
AVAILABLE_SECTIONS = [INTENT, SYNONYM, REGEX, LOOKUP]
MARKDOWN_SECTION_MARKERS = [f"## {s}:" for s in AVAILABLE_SECTIONS]

item_regex = re.compile(r"\s*[-*+]\s*((?:.+\s*)*)")
comment_regex = re.compile(r"<!--[\s\S]*?--!*>", re.MULTILINE)
fname_regex = re.compile(r"\s*([^-*+]+)")

logger = logging.getLogger(__name__)


class MarkdownReader(TrainingDataReader):
    """Reads markdown training data and creates a TrainingData object."""

    def __init__(self) -> None:
        super().__init__()
        self.current_title = None
        self.current_section = None
        self.training_examples = []
        self.entity_synonyms = {}
        self.regex_features = []
        self.lookup_tables = []

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Read markdown string and create TrainingData object"""
        s = self._strip_comments(s)
        for line in s.splitlines():
            line = decode_string(line.strip())
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
        import rasa.shared.nlu.training_data.lookup_tables_parser as lookup_tables_parser
        import rasa.shared.nlu.training_data.synonyms_parser as synonyms_parser
        from rasa.shared.nlu.training_data import entities_parser

        match = re.match(item_regex, line)
        if match:
            item = match.group(1)
            if self.current_section == INTENT:
                parsed = entities_parser.parse_training_example(
                    item, self.current_title
                )
                synonyms_parser.add_synonyms_from_entities(
                    parsed.get(TEXT), parsed.get("entities", []), self.entity_synonyms
                )
                self.training_examples.append(parsed)
            elif self.current_section == SYNONYM:
                synonyms_parser.add_synonym(
                    item, self.current_title, self.entity_synonyms
                )
            elif self.current_section == REGEX:
                self.regex_features.append(
                    {"name": self.current_title, "pattern": item}
                )
            elif self.current_section == LOOKUP:
                lookup_tables_parser.add_item_to_lookup_tables(
                    self.current_title, item, self.lookup_tables
                )

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
        import rasa.shared.utils.validation as validation_utils
        import rasa.shared.nlu.training_data.schemas.data_schema as schema

        # add {} as they are not part of the regex
        try:
            data = json.loads(f"{{{json_str}}}")
        except JSONDecodeError as e:
            rasa.shared.utils.io.raise_warning(
                f"Incorrect training data format ('{{{json_str}}}'), make sure your "
                f"data is valid. For more information about the format visit "
                f"{LEGACY_DOCS_BASE_URL}/nlu/training-data-format/."
            )
            raise e

        validation_utils.validate_training_data(data, schema.entity_dict_schema())

        return data

    def _set_current_section(self, section: Text, title: Text) -> None:
        """Update parsing mode."""
        if section not in AVAILABLE_SECTIONS:
            raise ValueError(
                "Found markdown section '{}' which is not "
                "in the allowed sections '{}'."
                "".format(section, "', '".join(AVAILABLE_SECTIONS))
            )

        self.current_section = section
        self.current_title = title

    @staticmethod
    def is_markdown_nlu_file(filename: Union[Text, Path]) -> bool:
        content = rasa.shared.utils.io.read_file(filename)
        return any(marker in content for marker in MARKDOWN_SECTION_MARKERS)


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

        import rasa.shared.nlu.training_data.util as rasa_nlu_training_data_utils

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
                self.generate_list_item(self.generate_message(example))
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

            md += self.generate_list_item(synonym[0])

        return md

    def _generate_regex_features_md(self, training_data: "TrainingData") -> Text:
        """Generates markdown for regex features."""

        md = ""
        # regex features are already sorted
        regex_features = training_data.regex_features
        for i, regex_feature in enumerate(regex_features):
            if i == 0 or regex_features[i - 1]["name"] != regex_feature["name"]:
                md += self._generate_section_header_md(REGEX, regex_feature["name"])

            md += self.generate_list_item(regex_feature["pattern"])

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
                    md += self.generate_list_item(e)
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
    def _generate_fname_md(text: Text) -> Text:
        """Generates markdown for a lookup table file path."""

        return f"  {encode_string(text)}\n"
