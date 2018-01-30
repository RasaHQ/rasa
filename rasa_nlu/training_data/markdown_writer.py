from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.utils import write_to_file
from rasa_nlu.training_data.markdown_reader import INTENT, SYNONYM, REGEX


class MarkdownWriter(object):

    def write(self, filename, training_data):
        """Writes a TrainingData object in markdown format to a file."""
        md = self.to_markdown(training_data)
        write_to_file(filename, md)

    def to_markdown(self, training_data):
        """Transforms a TrainingData object into a markdown string."""
        md = u''
        md += self._generate_training_examples_md(training_data)
        md += self._generate_synonyms_md(training_data)
        md += self._generate_regex_features_md(training_data)

        return md

    def _generate_training_examples_md(self, training_data):
        """generates markdown training examples."""
        training_examples = sorted([e.as_dict() for e in training_data.training_examples],
                                      key=lambda k: k['intent'])
        md = u''
        for i, example in enumerate(training_examples):
            if i == 0 or training_examples[i-1]['intent'] != example['intent']:
                md += self._create_section_header_text(INTENT, example['intent'], i != 0)

            md += self._generate_item_md(self._generate_message_md(example))

        return md

    def _generate_synonyms_md(self, training_data):
        """generates markdown for entity synomyms."""
        entity_synonyms = sorted(training_data.entity_synonyms.items(),
                                      key=lambda x: x[1])
        md = u''
        for i, synonym in enumerate(entity_synonyms):
            if i == 0 or entity_synonyms[i - 1][1] != synonym[1]:
                md += self._create_section_header_text(SYNONYM, synonym[1])

            md += self._generate_item_md(synonym[0])

        return md

    def _generate_regex_features_md(self, training_data):
        """generates markdown for regex features."""
        md = u''
        # regex features are already sorted
        regex_features = training_data.regex_features
        for i, regex_feature in enumerate(regex_features):
            if i == 0 or regex_features[i - 1]["name"] != regex_feature["name"]:
                md += self._create_section_header_text(REGEX, regex_feature["name"])

            md += self._generate_item_md(regex_feature["pattern"])

        return md

    def _create_section_header_text(self, section_type, title, prepend_newline = True):
        """generates markdown section header."""
        prefix = "\n" if prepend_newline else ""
        return prefix + "## {}:{}\n".format(section_type, title)

    def _generate_item_md(self, text):
        """generates markdown for a list item."""
        return "- {}\n".format(text)

    def _generate_message_md(self, message):
        """generates markdown for a message object."""
        md = ''
        text = message.get('text')
        entities = sorted(message.get('entities', []),
                          key=lambda k: k['start'])

        pos = 0
        for entity in entities:
            md += text[pos:entity['start']]
            md += self._generate_entity_md(text, entity)
            pos = entity['end']

        md += text[pos:]

        return md

    def _generate_entity_md(self, text, entity):
        """generates markdown for an entity object."""
        entity_text = text[entity['start']:entity['end']]
        entity_type = entity['entity']
        if entity_text != entity['value']:
            # add synonym suffix
            entity_type += ":{}".format(entity['value'])

        return '[{}]({})'.format(entity_text, entity_type)

