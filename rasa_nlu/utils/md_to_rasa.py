import codecs
import json
import sys
import re
import io
from rasa_nlu.training_data import Message

ent_regex = re.compile('\[(?P<value>[^\]]+)]\((?P<entity>[^:)]+)\)')  # [restaurant](what)
ent_regex_with_value = re.compile('\[(?P<synonym>[^\]]+)\]\((?P<entity>\w*?):(?P<value>[^)]+)\)')  # [open](open:1)
intent_regex = re.compile('##\s*intent:(.+)')
synonym_regex = re.compile('##\s*synonym:(.+)')
example_regex = re.compile('\s*-\s*(.+)')


class MarkdownToRasa:
    """ Converts training examples written in markdown to standard rasa json format """
    def __init__(self, file_name):
        self.file_name = file_name
        self.current_intent = None  # set when parsing examples from a given intent
        self.current_word = None  # set when parsing synonyms
        self.common_examples = []
        self.entity_synonyms = {}
        self.load()

    def get_example(self, utter):
        example = {'intent': self.current_intent}
        entities = []
        example['text'] = utter
        for regex in [ent_regex, ent_regex_with_value]:
            example['text'] = re.sub(regex, r"\1", example['text'])
            ent_matches = re.finditer(regex, utter)
            for matchNum, match in enumerate(ent_matches):
                entity_value_in_utter = match.groupdict()['synonym'] if 'synonym' in match.groupdict() \
                    else match.groupdict()['value']
                entities.append({
                    'entity': match.groupdict()['entity'],
                    'value': match.groupdict()['value'],
                    'start': example['text'].index(entity_value_in_utter),
                    'end': (example['text'].index(entity_value_in_utter) + len(entity_value_in_utter))
                })
        if len(entities) > 0:
            example['entities'] = entities
            return Message(example['text'], {'intent': example['intent'], 'entities': example['entities']})
        return Message(example['text'], {'intent':example['intent']})

    def set_current_state(self, state, value):
        """ switch between 'intent' and 'synonyms' mode """
        if state == 'intent':
            self.current_intent = value
            self.current_word = None
        elif state == 'synonym':
            self.current_intent = None
            self.current_word = value
        else:
            raise ValueError('State must be either \'intent\' or \'synonym\'')

    def get_current_state(self):
        """ informs whether whether we are currently loading intents or synonyms """
        if self.current_intent is None and self.current_word is not None:
            return 'synonym'
        elif self.current_word is None and self.current_intent is not None:
            return 'intent'
        else:
            raise ValueError(
                'Inconsistent state: one and only one of \'current_intent\' or \'current_word\' should be None')

    def load(self):
        """ parse the content of the actual .md file """
        with io.open(self.file_name, 'rU', encoding="utf-8-sig") as f:
            for row in f:
                intent_match = re.findall(intent_regex, row)
                if len(intent_match) > 0:
                    self.set_current_state('intent', intent_match[0])
                    continue

                synonym_match = re.findall(synonym_regex, row)
                if len(synonym_match) > 0:
                    self.set_current_state('synonym', synonym_match[0])
                    continue

                example_match = re.finditer(example_regex, row)
                for matchIndex, match in enumerate(example_match):
                    if self.get_current_state() == 'intent':
                        self.common_examples.append(self.get_example(match.group(1)))
                    else:
                        self.entity_synonyms[match.group(1)] = self.current_word
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