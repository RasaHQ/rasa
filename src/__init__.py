from pkg_resources import get_distribution
import codecs
import json
import os
import version

__version__ = version.__version__


class Interpreter(object):
    def parse(self, text):
        raise NotImplementedError()

    @staticmethod
    def load_synonyms(entity_synonyms_file):
        if entity_synonyms_file and os.path.isfile(entity_synonyms_file):
            with codecs.open(entity_synonyms_file, encoding='utf-8') as infile:
                return json.loads(infile.read())

    @staticmethod
    def replace_synonyms(entities, entity_synonyms):
        for i in range(len(entities)):
            entity_value = entities[i]["value"]
            if entity_value.lower() in entity_synonyms:
                entities[i]["value"] = entity_synonyms[entity_value.lower()]
