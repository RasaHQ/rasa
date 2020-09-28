import logging
import os
from typing import Any, Dict, Optional, Text, List, Tuple

import rasa.shared.nlu.training_data.util
from rasa.shared.constants import DOCS_BASE_URL
from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataReader
from rasa.shared.nlu.training_data.util import transform_entity_synonyms
import rasa.shared.utils.io

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

DOCS_URL_MIGRATE_GOOGLE = DOCS_BASE_URL + "/migrate-from/google-dialogflow-to-rasa/"

DIALOGFLOW_PACKAGE = "dialogflow_package"
DIALOGFLOW_AGENT = "dialogflow_agent"
DIALOGFLOW_INTENT = "dialogflow_intent"
DIALOGFLOW_INTENT_EXAMPLES = "dialogflow_intent_examples"
DIALOGFLOW_ENTITIES = "dialogflow_entities"
DIALOGFLOW_ENTITY_ENTRIES = "dialogflow_entity_entries"


class DialogflowReader(TrainingDataReader):
    def read(self, fn: Text, **kwargs: Any) -> "TrainingData":
        """Loads training data stored in the Dialogflow data format."""
        language = kwargs["language"]
        fformat = kwargs["fformat"]

        if fformat not in {DIALOGFLOW_INTENT, DIALOGFLOW_ENTITIES}:
            raise ValueError(
                "fformat must be either {}, or {}"
                "".format(DIALOGFLOW_INTENT, DIALOGFLOW_ENTITIES)
            )

        root_js = rasa.shared.utils.io.read_json_file(fn)
        examples_js = self._read_examples_js(fn, language, fformat)

        if not examples_js:
            rasa.shared.utils.io.raise_warning(
                f"No training examples found for dialogflow file {fn}!",
                docs=DOCS_URL_MIGRATE_GOOGLE,
            )
            return TrainingData()
        elif fformat == DIALOGFLOW_INTENT:
            return self._read_intent(root_js, examples_js)
        else:  # path for DIALOGFLOW_ENTITIES
            return self._read_entities(root_js, examples_js)

    def _read_intent(
        self, intent_js: Dict[Text, Any], examples_js: List[Dict[Text, Any]]
    ) -> "TrainingData":
        """Reads the intent and examples from respective jsons."""
        intent = intent_js.get("name")

        training_examples = []
        for ex in examples_js:
            text, entities = self._join_text_chunks(ex["data"])
            training_examples.append(Message.build(text, intent, entities))

        return TrainingData(training_examples)

    def _join_text_chunks(
        self, chunks: List[Dict[Text, Any]]
    ) -> Tuple[Text, List[Dict[Text, Any]]]:
        """Combines text chunks and extracts entities."""

        utterance = ""
        entities = []
        for chunk in chunks:
            entity = self._extract_entity(chunk, len(utterance))
            if entity:
                entities.append(entity)
            utterance += chunk["text"]

        return utterance, entities

    @staticmethod
    def _extract_entity(
        chunk: Dict[Text, Any], current_offset: int
    ) -> Optional[Dict[Text, Any]]:
        """Extract an entity from a chunk if present."""

        entity = None
        if "meta" in chunk or "alias" in chunk:
            start = current_offset
            text = chunk["text"]
            end = start + len(text)
            entity_type = chunk.get("alias", chunk["meta"])
            if entity_type != "@sys.ignore":
                entity = rasa.shared.nlu.training_data.util.build_entity(
                    start, end, text, entity_type
                )

        return entity

    @staticmethod
    def _flatten(list_of_lists: List[List[Any]]) -> List[Any]:
        return [item for items in list_of_lists for item in items]

    @staticmethod
    def _extract_lookup_tables(
        name: Text, examples: List[Dict[Text, Any]]
    ) -> Optional[List[Dict[Text, Any]]]:
        """Extract the lookup table from the entity synonyms"""
        synonyms = [e["synonyms"] for e in examples if "synonyms" in e]
        synonyms = DialogflowReader._flatten(synonyms)
        elements = [synonym for synonym in synonyms if "@" not in synonym]

        if len(elements) == 0:
            return None
        return [{"name": name, "elements": elements}]

    @staticmethod
    def _read_entities(entity_js, examples_js) -> "TrainingData":
        entity_synonyms = transform_entity_synonyms(examples_js)

        name = entity_js.get("name")
        lookup_tables = DialogflowReader._extract_lookup_tables(name, examples_js)
        return TrainingData([], entity_synonyms, [], lookup_tables)

    @staticmethod
    def _read_examples_js(fn: Text, language: Text, fformat: Text) -> Any:
        """Infer and load the example file based on the root
        filename and root format."""

        if fformat == DIALOGFLOW_INTENT:
            examples_type = "usersays"
        else:
            examples_type = "entries"
        examples_fn_ending = f"_{examples_type}_{language}.json"
        examples_fn = fn.replace(".json", examples_fn_ending)
        if os.path.isfile(examples_fn):
            return rasa.shared.utils.io.read_json_file(examples_fn)
        else:
            return None

    def reads(self, s, **kwargs):
        raise NotImplementedError
