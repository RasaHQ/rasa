from pkg_resources import get_distribution
import codecs
import json

__version__ = get_distribution('rasa_nlu').version


class Interpreter(object):
    def parse(self, text):
        raise NotImplementedError()

    @staticmethod
    def load_synonyms(entity_synonyms):
        if entity_synonyms:
            with codecs.open(entity_synonyms, encoding='utf-8') as infile:
                return json.loads(infile.read())

    @staticmethod
    def replace_synonyms(entities, entity_synonyms):
        for i in range(len(entities)):
            entity_value = entities[i]["value"]
            if entity_value.lower() in entity_synonyms:
                entities[i]["value"] = entity_synonyms[entity_value.lower()]
