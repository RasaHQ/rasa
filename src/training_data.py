# -*- coding: utf-8 -*-

import codecs
import json
import re
import warnings
from itertools import groupby

from rasa_nlu.utils import mitie

from rasa_nlu.utils import spacy

from rasa_nlu import util


# Different supported file formats and their identifier
WIT_FILE_FORMAT = "wit"
API_FILE_FORMAT = "api"
LUIS_FILE_FORMAT = "luis"
RASA_FILE_FORMAT = "rasa_nlu"
UNK_FILE_FORMAT = "unk"


class TrainingData(object):
    # Validation will ensure and warn if these lower limits are not met
    MIN_EXAMPLES_PER_INTENT = 2
    MIN_EXAMPLES_PER_ENTITY = 2

    def __init__(self, resource_name, backend, nlp=None, file_format=None):
        self.intent_examples = []
        self.entity_examples = []
        self.entity_synonyms = {}
        self.resource_name = resource_name
        self.files = TrainingData.resolve_data_files(resource_name)
        self.fformat = file_format if file_format is not None else TrainingData.guess_format(self.files)
        self.tokenizer = None

        self.init_tokenizer(backend, nlp)
        self.load_data()
        self.validate()

    def init_tokenizer(self, backend, nlp):
        if backend in [mitie.MITIE_BACKEND_NAME, mitie.MITIE_SKLEARN_BACKEND_NAME]:
            from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer
            self.tokenizer = MITIETokenizer()
        elif backend in [spacy.SPACY_BACKEND_NAME]:
            from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
            self.tokenizer = SpacyTokenizer(nlp)
        else:
            from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
            self.tokenizer = WhitespaceTokenizer()
            warnings.warn(
                "backend not recognised by TrainingData : defaulting to tokenizing by splitting on whitespace")

    @property
    def num_entity_examples(self):
        return len([e for e in self.entity_examples if len(e["entities"]) > 0])

    @staticmethod
    def resolve_data_files(resource_name):
        try:
            return util.recursively_find_files(resource_name)
        except ValueError as e:
            raise ValueError("Invalid training data file / folder specified. " + e.message)

    def as_json(self, **kwargs):
        return json.dumps({
            "rasa_nlu_data": {
                "intent_examples": self.intent_examples,
                "entity_examples": self.entity_examples
            }
        }, **kwargs)

    @staticmethod
    def guess_format(files):
        for filename in files:
            with codecs.open(filename, encoding="utf-8-sig") as f:
                file_data = json.loads(f.read())
            if "data" in file_data and type(file_data.get("data")) is list:
                return WIT_FILE_FORMAT
            elif "luis_schema_version" in file_data:
                return LUIS_FILE_FORMAT
            elif "userSays" in file_data:
                return API_FILE_FORMAT
            elif "rasa_nlu_data" in file_data:
                return RASA_FILE_FORMAT

        return UNK_FILE_FORMAT

    def load_data(self):
        if self.fformat == LUIS_FILE_FORMAT:
            self.load_luis_data(self.files[0])
        elif self.fformat == WIT_FILE_FORMAT:
            self.load_wit_data(self.files[0])
        elif self.fformat == API_FILE_FORMAT:
            self.load_api_data(self.files)
        elif self.fformat == RASA_FILE_FORMAT:
            self.load_rasa_data(self.files[0])
        else:
            raise ValueError("unknown training file format : {0}".format(self.fformat))

    def load_wit_data(self, filename):
        with codecs.open(filename, encoding="utf-8-sig") as f:
            data = json.loads(f.read())
        for s in data["data"]:
            entities = s.get("entities")
            if entities is None:
                continue
            text = s.get("text")
            intents = [e["value"] for e in entities if e["entity"] == 'intent']
            intent = intents[0] if intents else 'None'

            entities = [e for e in entities if ("start" in e and "end" in e)]
            for e in entities:
                e["value"] = e["value"][1:-1]
                # create synonyms dictionary
                text_value = text[e["start"]:e["end"]]
                util.add_entities_if_synonyms(self.entity_synonyms, text_value, e["value"])

            self.intent_examples.append({"text": text, "intent": intent})
            self.entity_examples.append({"text": text, "intent": intent, "entities": entities})

    def load_luis_data(self, filename):
        warnings.warn(
            """LUIS data may not always be correctly imported because entity locations are specified by tokens.
            If you use a tokenizer which behaves differently from LUIS's your entities might not be correct""")
        with codecs.open(filename, encoding="utf-8-sig") as f:
            data = json.loads(f.read())
        for s in data["utterances"]:
            text = s.get("text")
            tokens = [t for t in self.tokenizer.tokenize(text)]
            intent = s.get("intent")
            entities = []
            for e in s.get("entities") or []:
                i, ii = e["startPos"], e["endPos"] + 1
                _regex = u"\s*".join([re.escape(s) for s in tokens[i:ii]])
                expr = re.compile(_regex)
                m = expr.search(text)
                start, end = m.start(), m.end()
                val = text[start:end]
                entities.append({"entity": e["entity"], "value": val, "start": start, "end": end})

            self.intent_examples.append({"text": text, "intent": intent})
            self.entity_examples.append({"text": text, "intent": intent, "entities": entities})

    def load_api_data(self, files):
        for filename in files:
            with codecs.open(filename, encoding="utf-8-sig") as f:
                data = json.loads(f.read())
            # get only intents, skip the rest. The property name is the target class
            if "userSays" in data:
                intent = data.get("name")
                for s in data["userSays"]:
                    text = "".join(map(lambda chunk: chunk["text"], s.get("data")))
                    # add entities to each token, if available
                    entities = []
                    for e in filter(lambda chunk: "alias" in chunk or "meta" in chunk, s.get("data")):
                        start = text.find(e["text"])
                        end = start + len(e["text"])
                        val = text[start:end]
                        entities.append(
                            {
                                "entity": e["alias"] if "alias" in e else e["meta"],
                                "value": val,
                                "start": start,
                                "end": end
                            }
                        )

                    self.intent_examples.append({"text": text, "intent": intent})
                    self.entity_examples.append({"text": text, "intent": intent, "entities": entities})

            # create synonyms dictionary
            if "name" in data and "entries" in data:
                for entry in data["entries"]:
                    if "value" in entry and "synonyms" in entry:
                        for synonym in entry["synonyms"]:
                            util.add_entities_if_synonyms(self.entity_synonyms, synonym, entry["value"])

    def load_rasa_data(self, filename):
        with codecs.open(filename, encoding="utf-8-sig") as f:
            data = json.loads(f.read())
        common = data['rasa_nlu_data'].get("common_examples", list())
        intent = data['rasa_nlu_data'].get("intent_examples", list())
        entity = data['rasa_nlu_data'].get("entity_examples", list())

        self.intent_examples = intent + common
        self.entity_examples = entity + common

        for example in self.entity_examples:
            for entity in example["entities"]:
                entity_val = example["text"][entity["start"]:entity["end"]]
                util.add_entities_if_synonyms(self.entity_synonyms, entity_val, entity.get("value"))

    def sorted_entity_examples(self):
        return sorted([entity for ex in self.entity_examples for entity in ex["entities"]], key=lambda e: e["entity"])

    def sorted_intent_examples(self):
        return sorted(self.intent_examples, key=lambda e: e["intent"])

    def validate(self):
        examples = self.sorted_intent_examples()
        for intent, group in groupby(examples, lambda e: e["intent"]):
            size = len(list(group))
            if size < self.MIN_EXAMPLES_PER_INTENT:
                template = u"Intent '{0}' has only {1} training examples! minimum is {2}, training may fail."
                warnings.warn(template.format(intent, size, self.MIN_EXAMPLES_PER_INTENT))

        sorted_entity_examples = self.sorted_entity_examples()
        for entity, group in groupby(sorted_entity_examples, lambda e: e["entity"]):
            size = len(list(group))
            if size < self.MIN_EXAMPLES_PER_ENTITY:
                template = u"Entity '{0}' has only {1} training examples! minimum is {2}, training may fail."
                warnings.warn(template.format(entity, size, self.MIN_EXAMPLES_PER_ENTITY))

        for example in self.entity_examples:
            text = example["text"]
            text_tokens = self.tokenizer.tokenize(text)
            for ent in example["entities"]:
                ent_tokens = self.tokenizer.tokenize(text[ent["start"]:ent["end"]])
                for token in ent_tokens:
                    if token not in text_tokens:
                        warnings.warn(
                            "Token '{0}' does not appear in tokenized sentence {1}.".format(token, text_tokens) +
                            "Entities must span whole tokens.")
