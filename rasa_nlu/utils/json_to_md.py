from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class JsonToMd(object):
    def __init__(self, common_examples, entity_synonyms=None):
        self.common_examples = sorted([e.as_dict() for e in common_examples],
                                      key=lambda k: k['intent'])
        self.entity_synonyms = sorted(entity_synonyms.items(),
                                      key=lambda x: x[1])

    def to_markdown(self):
        output = u''
        previous_intent = None
        for i, example in enumerate(self.common_examples):
            if previous_intent != example['intent']:
                if previous_intent is not None:
                    output += '\n'
                output += '## intent:{}\n'.format(example['intent'])

            output += '- {}\n'.format(self.example_to_md(example))
            previous_intent = example['intent']

        for i, synonym in enumerate(self.entity_synonyms):
            if i == 0 or self.entity_synonyms[i - 1][1] != synonym[1]:
                output += '\n## synonym:{}\n'.format(synonym[1])

            output += '- {}\n'.format(synonym[0])
        return output

    def example_to_md(self, example):
        md_example = ''
        text = example.get('text')
        entities = sorted(example.get('entities', []),
                          key=lambda k: k['start'])
        position_pointer = 0

        for entity in entities:
            entity_value = text[entity['start']:entity['end']]
            md_example += text[position_pointer:entity['start']]
            md_example += '[{}]({})'.format(entity_value,
                                            self.get_entity_name(text, entity))
            position_pointer = entity['end']
        md_example += text[position_pointer:]

        return md_example

    @staticmethod
    def get_entity_name(text, entity):
        if entity['value'] == text[entity['start']:entity['end']]:
            return entity['entity']
        else:
            return "{}:{}".format(entity['entity'], entity['value'])
