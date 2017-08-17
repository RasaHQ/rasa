from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import io
from rasa_nlu.training_data import Message

ent_regex = re.compile('\[(?P<value>[^\]]+)]\((?P<entity>[^:)]+)\)')  # [restaurant](what)
ent_regex_with_value = re.compile('\[(?P<synonym>[^\]]+)\]\((?P<entity>\w*?):(?P<value>[^)]+)\)')  # [open](open:1)
intent_regex = re.compile('##\s*intent:(.+)')
synonym_regex = re.compile('##\s*synonym:(.+)')
example_regex = re.compile('\s*-\s*(.+)')


class MarkdownToJson:
    """ Converts training examples written in markdown to standard rasa json format """
    def __init__(self, file_name):
        self.file_name = file_name
        self.current_intent = None  # set when parsing examples from a given intent
        self.common_examples = []
        self.entity_synonyms = []
        self.load()

    def get_example(self, example_in_md):
        entities = []
        utter = example_in_md
        for regex in [ent_regex, ent_regex_with_value]:
            utter = re.sub(regex, r"\1", utter)  # [text](entity) -> text
            ent_matches = re.finditer(regex, example_in_md)
            for matchNum, match in enumerate(ent_matches):
                if 'synonym' in match.groupdict():
                    entity_value_in_utter = match.groupdict()['synonym']
                else:
                    entity_value_in_utter = match.groupdict()['value']

                entities.append({
                    'entity': match.groupdict()['entity'],
                    'value': match.groupdict()['value'],
                    'start': utter.index(entity_value_in_utter),
                    'end': (utter.index(entity_value_in_utter) + len(entity_value_in_utter))
                })

        message = Message(utter, {'intent': self.current_intent})
        if len(entities) > 0:
            message.set('entities', entities)
        return message

    def set_current_state(self, state, value):
        """ switch between 'intent' and 'synonyms' mode """
        if state == 'intent':
            self.current_intent = value
        elif state == 'synonym':
            self.current_intent = None
            self.entity_synonyms.append({'value': value, 'synonyms': []})
        else:
            raise ValueError("State must be either 'intent' or 'synonym'")

    def get_current_state(self):
        """ informs whether whether we are currently loading intents or synonyms """
        if self.current_intent is not None:
            return 'intent'
        else:
            return 'synonym'

    def load(self):
        """ parse the content of the actual .md file """
        with io.open(self.file_name, 'rU', encoding="utf-8-sig") as f:
            for row in f:
                intent_match = re.search(intent_regex, row)
                if intent_match is not None:
                    self.set_current_state('intent', intent_match.group(1))
                    continue

                synonym_match = re.search(synonym_regex, row)
                if synonym_match is not None:
                    self.set_current_state('synonym', synonym_match.group(1))
                    continue

                example_match = re.finditer(example_regex, row)
                for matchIndex, match in enumerate(example_match):
                    if self.get_current_state() == 'intent':
                        self.common_examples.append(self.get_example(match.group(1)))
                    else:
                        self.entity_synonyms[-1]['synonyms'].append(match.group(1))
        return {
            "rasa_nlu_data": {
                "common_examples": self.common_examples,
                "entity_synonyms": self.entity_synonyms
            }
        }

    def get_common_examples(self):
        return self.common_examples

    def get_entity_synonyms(self):
        return self.entity_synonyms
