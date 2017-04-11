from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component

DUCKLING_PROCESSING_MODES = ["replace", "append"]


class DucklingExtractor(Component):
    """Adds entity normalization by analyzing found entities and transforming them into regular formats."""

    name = "ner_duckling"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    def __init__(self, duckling=None):
        # type: (Optional[DucklingWrapper]) -> None
        from duckling import DucklingWrapper

        self.duckling = duckling

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text
        from rasa_nlu.model import Metadata

        return cls.name + "-" + model_metadata.language

    def pipeline_init(self, language, duckling_processing_mode):
        # type: (Text, Text) -> None
        from duckling import DucklingWrapper

        if duckling_processing_mode not in DUCKLING_PROCESSING_MODES:
            raise ValueError("Invalid duckling processing mode. Got '{}'. Allowed: {}".format(
                duckling_processing_mode, ", ".join(DUCKLING_PROCESSING_MODES)))

        if self.duckling is None:
            try:
                self.duckling = DucklingWrapper(language=language)  # languages in duckling are eg "de$core"
            except ValueError as e:
                raise Exception("Duckling error. {}".format(e.message))

    def process(self, text, entities, duckling_processing_mode):
        # type: (Text, List[Dict[Text, Any]], Text) -> Dict[Text, Any]

        if self.duckling is not None:
            parsed = self.duckling.parse(text)
            for duckling_match in parsed:
                for entity in entities:
                    if entity["start"] == duckling_match["start"] and entity["end"] == duckling_match["end"]:
                        entity["value"] = duckling_match["value"]["value"]
                        entity["duckling"] = duckling_match["dim"]
                        break
                else:
                    if duckling_processing_mode == "append":
                        # Duckling will retrieve multiple entities, even if they overlap..
                        # hence the append mode might add some noise to the found entities
                        entities.append({
                            "entity": duckling_match["dim"],
                            "duckling": duckling_match["dim"],
                            "value": duckling_match["value"]["value"],
                            "start": duckling_match["start"],
                            "end": duckling_match["end"],
                        })

        return {
            "entities": entities
        }
