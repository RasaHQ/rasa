import typing
from typing import Any, Dict, List, Text

from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.training_data import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyEntityExtractor(EntityExtractor):

    provides = ["entities"]

    requires = ["spacy_nlp"]

    defaults = {
        # by default all dimensions recognized by spacy are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None
    }

    def __init__(self, component_config: Text = None) -> None:
        super(SpacyEntityExtractor, self).__init__(component_config)

    def process(self, message: Message, **kwargs: Any) -> None:
        # can't use the existing doc here (spacy_doc on the message)
        # because tokens are lower cased which is bad for NER
        spacy_nlp = kwargs.get("spacy_nlp", None)
        doc = spacy_nlp(message.text)
        all_extracted = self.add_extractor_name(self.extract_entities(doc))
        dimensions = self.component_config["dimensions"]
        extracted = SpacyEntityExtractor.filter_irrelevant_entities(
            all_extracted, dimensions
        )
        message.set(
            "entities", message.get("entities", []) + extracted, add_to_output=True
        )

    @staticmethod
    def extract_entities(doc: "Doc") -> List[Dict[Text, Any]]:
        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "confidence": None,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return entities
