from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import io
import logging

from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
available_sections = [INTENT, SYNONYM, REGEX]
ent_regex = re.compile(r'\[(?P<entity_text>[^\]]+)'
                       r'\]\((?P<entity>\w*?)'
                       r'(?:\:(?P<value>[^)]+))?\)')  # [open](open:1)

example_regex = re.compile(r'\s*[-\*+]\s*(.+)')
comment_regex = re.compile(r'<!--[\s\S]*?--!*>', re.MULTILINE)

logger = logging.getLogger(__name__)

def create_section_regexes(section_names):
    def make_regex(section_name):
        return re.compile(r'##\s*{}:(.+)'.format(section_name))

    return {sn: make_regex(sn) for sn in section_names}

section_regexes = create_section_regexes(available_sections)

def strip_comments(comment_regex, text):
    """ Removes comments defined by `comment_regex` from `text`. """
    text = re.sub(comment_regex, '', text)
    text = text.splitlines()  # Split into lines
    return text


class MarkdownReader(object):
    """Reads markdown training data and creates a TrainingData object."""

    def __init__(self):
        # set when parsing examples from a given intent
        self.file_name = None
        self.current_title = None
        self.current_section = None
        self.common_examples = []
        self.entity_synonyms = {}
        self.regex_features = []

    def read(self, file_name):
        """Parse the content of the .md file."""
        self.__init__()
        self.file_name = file_name
        with io.open(file_name, 'rU', encoding="utf-8-sig") as f:
            f_com_rmved = strip_comments(comment_regex, f.read())  # Strip comments
            for line in f_com_rmved:
                # Remove white-space which may have crept in due to comments
                line = line.strip()
                header = self._find_section_header(line)
                if header:
                    self._set_current_section(header[0], header[1])
                else:
                    self._parse_example(line)

        return TrainingData(self.common_examples, self.entity_synonyms, self.regex_features)

    def _find_section_header(self, line):
        """Checks if the current line contains a section header and returns the section and the title."""
        for name, regex in section_regexes.items():
            match = re.search(regex, line)
            if match is not None:
                return name, match.group(1)
        return None

    def _parse_example(self, row):
        """Parses an md row based on the current section type."""
        example_match = re.finditer(example_regex, row)
        for matchIndex, match in enumerate(example_match):
            example = match.group(1)
            if self.current_section == INTENT:
                parsed = self._parse_intent_example(example)
                self.common_examples.append(parsed)
            elif self.current_section == SYNONYM:
                self._add_synonym(example, self.current_title)
            else:
                self.regex_features.append({"name": self.current_title, "pattern": example})

    def _find_entities_in_intent_example(self, example):
        """Extracts entities from a markdown intent example."""
        entities = []
        offset = 0
        for match in re.finditer(ent_regex, example):
            entity_text = match.groupdict()['entity_text']
            entity_entity = match.groupdict()['entity']
            entity_value = match.groupdict()['value'] if match.groupdict()['value'] else entity_text

            start_index = match.start() - offset
            end_index = start_index + len(entity_text)
            offset += len(match.group(0)) - len(entity_text)

            entities.append({
                'entity': entity_entity,
                'value': entity_value,
                'start': start_index,
                'end': end_index
            })

        return entities

    def _add_synonym(self, text, value):
        if text in self.entity_synonyms and self.entity_synonyms[text] != value:
            logger.warning("Inconsistent entity synonyms in file {0}, overwriting {1}->{2}"
                           "with {1}->{3}".format(self.file_name, text, self.entity_synonyms[text], value))
        self.entity_synonyms[text] = value

    def _add_synonyms(self, plain_text, entities):
        """Adds synonyms found in intent examples"""
        for e in entities:
            e_text = plain_text[e['start']:e['end']]
            if e_text != e['value']:
                self._add_synonym(e_text, e['value'])

    def _parse_intent_example(self, example):
        """Extract entities and synonyms, and convert to plain text."""
        entities = self._find_entities_in_intent_example(example)
        plain_text = re.sub(ent_regex, lambda m: m.groupdict()['entity_text'], example)
        self._add_synonyms(plain_text, entities)
        message = Message(plain_text, {'intent': self.current_title})
        if len(entities) > 0:
            message.set('entities', entities)
        return message

    def _set_current_section(self, section, title):
        """Update parsing mode."""
        if section not in available_sections:
            raise ValueError("Found markdown section {} which is not "
                             "in the allowed sections {},".format(section, ",".join(available_sections)))

        self.current_section = section
        self.current_title = title
