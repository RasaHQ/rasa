import typing
from typing import Any, Dict, List, Text

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyEntityExtractor(EntityExtractor):

    provides = ["entities"]

    requires = ["spacy_nlp"]

    def process(self, message: Message, **kwargs: Any) -> None:
        # can't use the existing doc here (spacy_doc on the message)
        # because tokens are lower cased which is bad for NER
        spacy_nlp = kwargs.get("spacy_nlp", None)
        doc = spacy_nlp(message.text)
        extracted = self.add_extractor_name(self.extract_entities(doc))
        message.set("entities",
                    message.get("entities", []) + extracted,
                    add_to_output=True)

    @staticmethod
    def extract_entities(doc: 'Doc') -> List[Dict[Text, Any]]:
        entities = [
            {
                "entity": ent.label_,
                "value": ent.text,
                "start": ent.start_char,
                "confidence": None,
                "end": ent.end_char
            }
            for ent in doc.ents]
        return entities
