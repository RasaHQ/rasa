from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import io
from rasa_nlu.training_data import Message
from collections import defaultdict

INTENT = "intent"
SYNONYM = "synonym"
REGEX = "regex"
available_sections = [INTENT, SYNONYM, REGEX]
ent_regex = re.compile(r'\[(?P<entity_text>[^\]]+)'
                       r'\]\((?P<entity>\w*?)'
                       r'(?:\:(?P<value>[^)]+))?\)')  # [open](open:1)

example_regex = re.compile(r'\s*[-\*+]\s*(.+)')
comment_regex = re.compile(r'<!--[\s\S]*?--!*>', re.MULTILINE)



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


class MarkdownToJson(object):
    """Converts training examples written in md to standard rasa json format."""

    def __init__(self, file_name):
        self.file_name = file_name
        # set when parsing examples from a given intent
        self.current_title = None
        self.current_section = None
        self.common_examples = []
        self.entity_synonyms = defaultdict(set)
        self.regex_patterns = []

    def find_section_header(self, line):
        """Checks if the current line contains a section header and returns the section and the title."""
        for name, regex in section_regexes.items():
            match = re.search(regex, line)
            if match is not None:
                return name, match.group(1)
        return None

    def make_json(self):
        """Combines the parsed data into the json training data format."""
        return {
            "rasa_nlu_data": {
                "common_examples": self.common_examples,
                "entity_synonyms": [{"value": val, "synonyms": list(syns)}
                                    for val, syns in self.entity_synonyms.items()],
                "regex_features": self.regex_patterns
            }
        }

    def load(self):
        """Parse the content of the actual .md file."""
        with io.open(self.file_name, 'rU', encoding="utf-8-sig") as f:
            f_com_rmved = strip_comments(comment_regex, f.read())  # Strip comments
            for line in f_com_rmved:
                # Remove white-space which may have crept in due to comments
                line = line.strip()
                header = self.find_section_header(line)
                if header:
                    self._set_current_section(header[0], header[1])
                else:
                    self._parse_example(line)

        return self.make_json()

    def _parse_example(self, row):
        """Parses an md row based on the current section type."""
        example_match = re.finditer(example_regex, row)
        for matchIndex, match in enumerate(example_match):
            example = match.group(1)
            if self.current_section == INTENT:
                parsed = self._parse_intent_example(example)
                self.common_examples.append(parsed)
            elif self.current_section == SYNONYM:
                self.entity_synonyms[self.current_title].add(example)
            else:
                self.regex_patterns.append({"name": self.current_title, "pattern": example})

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

    def _add_synonyms(self, plain_text, entities):
        """Adds synonyms found in intent examples"""
        for e in entities:
            e_text = plain_text[e['start']:e['end']]
            if e_text != e['value']:
                self.entity_synonyms[e['value']].add(e_text)

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
