from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyDependancyExtractor(EntityExtractor):
    name = "dep_spacy"

    provides = ["entities"]

    requires = ["spacy_nlp"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated_entities = message.get("entities", [])[:]
        doc = message.get("spacy_doc")

        self.parse_dependencies(doc, updated_entities)
        message.set("entities", updated_entities, add_to_output=True)

    def parse_dependencies(self, doc, entities):
        for token in doc:
            for entity in entities:
                entity_value = str(entity["value"]).lower()
                token_found = entity_value.find(token.text) != -1
                token_ad = token.head.pos_ == 'ADP'

                if token_found and token_ad:
                    entity["adposition"] = token.head.text
                    self.add_processor_name(entity)