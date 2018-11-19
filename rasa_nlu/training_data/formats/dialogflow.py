from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from rasa_nlu.training_data import Message, TrainingData
from rasa_nlu.training_data.formats.readerwriter import TrainingDataReader
from rasa_nlu import utils
from rasa_nlu.training_data.util import transform_entity_synonyms

logger = logging.getLogger(__name__)

DIALOGFLOW_PACKAGE = "dialogflow_package"
DIALOGFLOW_AGENT = "dialogflow_agent"
DIALOGFLOW_INTENT = "dialogflow_intent"
DIALOGFLOW_INTENT_EXAMPLES = "dialogflow_intent_examples"
DIALOGFLOW_ENTITIES = "dialogflow_entities"
DIALOGFLOW_ENTITY_ENTRIES = "dialogflow_entity_entries"


class DialogflowReader(TrainingDataReader):
    def read(self, fn, **kwargs):
        # type: ([Text]) -> TrainingData
        """Loads training data stored in the Dialogflow data format."""

        language = kwargs["language"]
        fformat = kwargs["fformat"]

        if fformat not in {DIALOGFLOW_INTENT, DIALOGFLOW_ENTITIES}:
            raise ValueError("fformat must be either {}, or {}".format(DIALOGFLOW_INTENT, DIALOGFLOW_ENTITIES))

        root_js = utils.read_json_file(fn)
        examples_js = self._read_examples_js(fn, language, fformat)

        if not examples_js:
            logger.warning("No training examples found for dialogflow file {}!".format(fn))
            return TrainingData()
        elif fformat == DIALOGFLOW_INTENT:
            return self._read_intent(root_js, examples_js)
        elif fformat == DIALOGFLOW_ENTITIES:
            return self._read_entities(root_js, examples_js)

    def _get_intent_name(self, intent_js):
        intent_name = None
        if(intent_js.get("responses") and intent_js.get("responses")[0]):
            intent_name = intent_js.get("responses")[0].get('action')
        if(intent_name is None):
            intent_name = intent_js.get("name")
        return intent_name

    def _read_intent(self, intent_js, examples_js):
        """Reads the intent and examples from respective jsons."""
        intent = self._get_intent_name(intent_js)

        training_examples = []
        for ex in examples_js:
            text, entities = self._join_text_chunks(ex['data'])
            training_examples.append(Message.build(text, intent, entities))

        return TrainingData(training_examples)

    def _join_text_chunks(self, chunks):
        """Combines text chunks and extracts entities."""
        utterance = ""
        entities = []
        for chunk in chunks:
            entity = self._extract_entity(chunk, len(utterance))
            if entity:
                entities.append(entity)
            utterance += chunk["text"]

        return utterance, entities

    def _extract_entity(self, chunk, current_offset):
        """Extract an entity from a chunk if present."""
        entity = None
        if "meta" in chunk or "alias" in chunk:
            start = current_offset
            text = chunk['text']
            end = start + len(text)
            entity_type = chunk.get("alias", chunk["meta"])
            if entity_type != u'@sys.ignore':
                entity = utils.build_entity(start, end, text, entity_type)

        return entity

    def _flatten(self, list_of_lists):
        return [item for items in list_of_lists for item in items]

    def _extract_lookup_tables(self, entity, examples):
        """Extract the lookup table from the entity synonyms"""
        lookup_tables = []
        synonyms = [e["synonyms"] for e in examples if "synonyms" in e]
        synonyms = self._flatten(synonyms)
        for synonym in synonyms:
            if "@" not in synonym:
                lookup_tables.append(synonym)
        if len(lookup_tables) == 0:
            return False
        return [{
            'name': entity,
            'elements': lookup_tables
        }]

    def _add_to_composites(self, each, composite_entities):
        if each:
            if(each[0:11] == '@sys.number'):
                composite_entities.add("@" + each[12:])
            else:
                composite_entities.add(each.split(':')[0])
        return composite_entities

    def _extract_composite_entities(self, entity, synonyms):
        """Extract the composite entities"""
        composite_entities = set()
        words = [s["value"].split(" ") for s in
                synonyms if "value" in s and "@" in s["value"]]
        words = self._flatten(words)
        for word in words:
            composite_entities = self._add_to_composites(
                                    word,
                                    composite_entities)
        if len(composite_entities) == 0:
            return False
        return [{
            'name': entity,
            'composites': list(composite_entities)
        }]

    def _read_entities(self, entity_js, examples_js):
        entity = entity_js.get("name")
        entity_synonyms = transform_entity_synonyms(examples_js)
        lookup_tables = self._extract_lookup_tables(entity, examples_js)
        composite_entities = self._extract_composite_entities(
                                entity,
                                examples_js)
        return TrainingData([],
                            entity_synonyms,
                            [],
                            lookup_tables,
                            composite_entities)

    def _read_examples_js(self, fn, language, fformat):
        """Infer and load the example file based on the root filename and root format."""
        examples_type = "usersays" if fformat == DIALOGFLOW_INTENT else "entries"
        examples_fn_ending = "_{}_{}.json".format(examples_type, language)
        examples_fn = fn.replace(".json", examples_fn_ending)
        if os.path.isfile(examples_fn):
            return utils.read_json_file(examples_fn)
        else:
            return None

    def reads(self, s, **kwargs):
        raise NotImplementedError
