class JsonToMd:

    def __init__(self, common_examples, entity_synonyms=None):
        self.common_examples = sorted([e.as_dict() for e in common_examples], key=lambda k: k['intent'])
        self.entity_synonyms = sorted(entity_synonyms.items(), key=lambda x: x[1])

    def to_markdown(self):
        output = u''
        for i, example in enumerate(self.common_examples):
            if i > 0 and self.common_examples[i - 1]['intent'] != example['intent']:
                output += '\n'

            if i == 0 or self.common_examples[i - 1]['intent'] != example['intent']:
                output += '## intent:{}\n'.format(example['intent'])

            output += '- {}\n'.format(self.example_to_md(example))

        for i, synonym in enumerate(self.entity_synonyms):
            if i == 0 or self.entity_synonyms[i-1][1] != synonym[1]:
                output += '\n## synonym:{}\n'.format(synonym[1])

            output += '- {}\n'.format(synonym[0])
        return output

    def example_to_md(self, example):
        md_example = ''
        if 'entities' in example is not None and len(example['entities']) > 0:
            entities = sorted(example['entities'], key=lambda k: k['start'])
            position_pointer = 0
            for entity in entities:
                md_example += example['text'][position_pointer:entity['start']]
                md_example += '[{}]({})'.format(example['text'][entity['start']:entity['end']],
                                                self.get_entity_name(example['text'], entity))
                position_pointer = entity['end']
            md_example += example['text'][position_pointer:]
        else:
            md_example = example['text']

        return md_example

    @staticmethod
    def get_entity_name(text, entity):
        if entity['value'] == text[entity['start']:entity['end']]:
            return entity['entity']
        else:
            return "{}:{}".format(entity['entity'], entity['value'])
