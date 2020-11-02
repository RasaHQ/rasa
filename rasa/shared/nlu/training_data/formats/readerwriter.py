import json
from collections import OrderedDict
from pathlib import Path

import rasa.shared.nlu.training_data.util
from rasa.shared.constants import INTENT_MESSAGE_PREFIX

from rasa.shared.nlu.constants import (
    INTENT,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
)

import rasa.shared.utils.io
import typing
from typing import Text, Dict, Any, Union

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.training_data import TrainingData


class TrainingDataReader:
    def __init__(self):
        self.filename: Text = ""

    def read(self, filename: Union[Text, Path], **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a file."""
        self.filename = filename
        return self.reads(rasa.shared.utils.io.read_file(filename), **kwargs)

    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a string."""
        raise NotImplementedError


class TrainingDataWriter:
    def dump(self, filename: Text, training_data) -> None:
        """Writes a TrainingData object in markdown format to a file."""
        s = self.dumps(training_data)
        rasa.shared.utils.io.write_text_file(s, filename)

    def dumps(self, training_data: "TrainingData") -> Text:
        """Turns TrainingData into a string."""
        raise NotImplementedError

    @staticmethod
    def prepare_training_examples(training_data: "TrainingData") -> OrderedDict:
        """Pre-processes training data examples by removing not trainable entities."""

        import rasa.shared.nlu.training_data.util as rasa_nlu_training_data_utils

        training_examples = OrderedDict()

        # Sort by intent while keeping basic intent order
        for example in [e.as_dict_nlu() for e in training_data.training_examples]:
            if not example.get(INTENT):
                continue
            rasa_nlu_training_data_utils.remove_untrainable_entities_from(example)
            intent = example[INTENT]
            training_examples.setdefault(intent, [])
            training_examples[intent].append(example)

        return training_examples

    @staticmethod
    def generate_list_item(text: Text) -> Text:
        """Generates text for a list item."""

        return f"- {rasa.shared.nlu.training_data.util.encode_string(text)}\n"

    @staticmethod
    def generate_message(message: Dict[Text, Any]) -> Text:
        """Generates text for a message object."""

        md = ""
        text = message.get("text", "")

        pos = 0

        # If a message was prefixed with `INTENT_MESSAGE_PREFIX` (this can only happen
        # in end-to-end stories) then potential entities were provided in the json
        # format (e.g. `/greet{"name": "Rasa"}) and we don't have to add the NLU
        # entity annotation
        if not text.startswith(INTENT_MESSAGE_PREFIX):
            entities = sorted(message.get("entities", []), key=lambda k: k["start"])

            for entity in entities:
                md += text[pos : entity["start"]]
                md += TrainingDataWriter.generate_entity(text, entity)
                pos = entity["end"]

        md += text[pos:]

        return md

    @staticmethod
    def generate_entity(text: Text, entity: Dict[Text, Any]) -> Text:
        """Generates text for an entity object."""

        entity_text = text[
            entity[ENTITY_ATTRIBUTE_START] : entity[ENTITY_ATTRIBUTE_END]
        ]
        entity_type = entity.get(ENTITY_ATTRIBUTE_TYPE)
        entity_value = entity.get(ENTITY_ATTRIBUTE_VALUE)
        entity_role = entity.get(ENTITY_ATTRIBUTE_ROLE)
        entity_group = entity.get(ENTITY_ATTRIBUTE_GROUP)

        if entity_value and entity_value == entity_text:
            entity_value = None

        use_short_syntax = (
            entity_value is None and entity_role is None and entity_group is None
        )

        if use_short_syntax:
            return f"[{entity_text}]({entity_type})"
        else:
            entity_dict = OrderedDict(
                [
                    (ENTITY_ATTRIBUTE_TYPE, entity_type),
                    (ENTITY_ATTRIBUTE_ROLE, entity_role),
                    (ENTITY_ATTRIBUTE_GROUP, entity_group),
                    (ENTITY_ATTRIBUTE_VALUE, entity_value),
                ]
            )
            entity_dict = OrderedDict(
                [(k, v) for k, v in entity_dict.items() if v is not None]
            )

            return f"[{entity_text}]{json.dumps(entity_dict)}"


class JsonTrainingDataReader(TrainingDataReader):
    def reads(self, s: Text, **kwargs: Any) -> "TrainingData":
        """Transforms string into json object and passes it on."""
        js = json.loads(s)
        return self.read_from_json(js, **kwargs)

    def read_from_json(self, js: Dict[Text, Any], **kwargs: Any) -> "TrainingData":
        """Reads TrainingData from a json object."""
        raise NotImplementedError
