from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals, print_function

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.extractors import EntityExtractor

if typing.TYPE_CHECKING:
    from spacy.language import Language


class SpacyEntityExtractor(EntityExtractor):
    name = "ner_spacy"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["spacy"]

    def process(self, text, spacy_nlp, entities):
        # type: (Doc, Language, List[Dict[Text, Any]]) -> Dict[Text, Any]
        extracted = self.add_extractor_name(self.extract_entities(text, spacy_nlp))
        entities.extend(extracted)
        return {
            "entities": entities
        }

    def extract_entities(self, text, spacy_nlp):
        # type: (Text, Language) -> List[Dict[Text, Any]]

        if spacy_nlp.entity is not None:
            doc = spacy_nlp(text)

            entities = [
                {
                    "entity": ent.label_,
                    "value": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents]
            return entities
        else:
            return []
