from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import io
from rasa_nlu.training_data import Message

ent_regex = re.compile(r'\[(?P<synonym>[^\]]+)'
                       r'\]\((?P<entity>\w*?)'
                       r'(?:\:(?P<value>[^)]+))?\)')  # [open](open:1)
intent_regex = re.compile(r'##\s*intent:(.+)')
synonym_regex = re.compile(r'##\s*synonym:(.+)')
example_regex = re.compile(r'\s*[-\*]\s*(.+)')
comment_regex = re.compile(r'<!--[\s\S]*?--!*>', re.MULTILINE)

INTENT_PARSING_STATE = "intent"
SYNONYM_PARSING_STATE = "synonym"


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
        self.current_intent = None
        self.common_examples = []
        self.entity_synonyms = []
        self.load()

    def load(self):
        """Parse the content of the actual .md file."""

        with io.open(self.file_name, 'rU', encoding="utf-8-sig") as f:
            f_com_rmved = strip_comments(comment_regex, f.read())  # Strip comments
            for row in f_com_rmved:
                # Remove white-space which may have crept in due to comments
                row = row.strip()
                intent_match = re.search(intent_regex, row)
                if intent_match is not None:
                    self._set_current_state(
                            INTENT_PARSING_STATE, intent_match.group(1))
                    continue

                synonym_match = re.search(synonym_regex, row)
                if synonym_match is not None:
                    self._set_current_state(
                            SYNONYM_PARSING_STATE, synonym_match.group(1))
                    continue

                self._parse_intent_or_synonym_example(row)
        return {
            "rasa_nlu_data": {
                "common_examples": self.common_examples,
                "entity_synonyms": self.entity_synonyms
            }
        }

    def _parse_intent_or_synonym_example(self, row):
        example_match = re.finditer(example_regex, row)
        for matchIndex, match in enumerate(example_match):
            example_line = match.group(1)
            if self._current_state() == INTENT_PARSING_STATE:
                parsed = self._parse_intent_example(example_line)
                self.common_examples.append(parsed)
            else:
                self.entity_synonyms[-1]['synonyms'].append(example_line)

    def _parse_intent_example(self, example_in_md):
        entities = []
        utter = example_in_md
        match = re.search(ent_regex, utter)
        while match is not None:
            entity_synonym = match.groupdict()['synonym']
            entity_entity = match.groupdict()['entity']
            entity_value = match.groupdict()['value']

            if match.groupdict()['value'] is None:
                entity_value = entity_synonym

            start_index = match.start()
            end_index = start_index + len(entity_synonym)

            entities.append({
                'entity': entity_entity,
                'value': entity_value,
                'start': start_index,
                'end': end_index
            })

            utter = utter[:match.start()] + entity_synonym + utter[match.end():]
            match = re.search(ent_regex, utter)

        message = Message(utter, {'intent': self.current_intent})
        if len(entities) > 0:
            message.set('entities', entities)
        return message

    def _set_current_state(self, state, value):
        """Switch between 'intent' and 'synonyms' mode."""

        if state == INTENT_PARSING_STATE:
            self.current_intent = value
        elif state == SYNONYM_PARSING_STATE:
            self.current_intent = None
            self.entity_synonyms.append({'value': value, 'synonyms': []})
        else:
            raise ValueError("State must be either '{}' or '{}'".format(
                    INTENT_PARSING_STATE, SYNONYM_PARSING_STATE))

    def _current_state(self):
        """Informs whether we are currently loading intents or synonyms."""

        if self.current_intent is not None:
            return INTENT_PARSING_STATE
        else:
            return SYNONYM_PARSING_STATE
