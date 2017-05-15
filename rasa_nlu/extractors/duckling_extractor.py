from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import io
import json
import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from builtins import str

from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from inspect import getmembers

if typing.TYPE_CHECKING:
    from duckling import DucklingWrapper


class DucklingExtractor(EntityExtractor):
    """Adds entity normalization by analyzing found entities and transforming them into regular formats."""

    name = "ner_duckling"

    context_provides = {
        "process": ["entities"],
    }

    output_provides = ["entities"]

    @staticmethod
    def available_dimensions():
        from duckling.dim import Dim
        return [m[1] for m in getmembers(Dim) if not m[0].startswith("__") and not m[0].endswith("__")]

    def __init__(self, dimensions=None, duckling=None):
        # type: (Text, Optional[DucklingWrapper]) -> None

        self.dimensions = dimensions if dimensions else self.available_dimensions()
        self.duckling = duckling

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["duckling"]

    @classmethod
    def create(cls, duckling_dimensions):
        if duckling_dimensions is None:
            duckling_dimensions = cls.available_dimensions()
        unknown_dimensions = [dim for dim in duckling_dimensions if dim not in cls.available_dimensions()]
        if len(unknown_dimensions) > 0:
            raise ValueError("Invalid duckling dimension. Got '{}'. Allowed: {}".format(
                ", ".join(unknown_dimensions), ", ".join(cls.available_dimensions())))

        return DucklingExtractor(duckling_dimensions)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        return cls.name + "-" + model_metadata.language

    def pipeline_init(self, language):
        # type: (Text, Text) -> None
        from duckling import DucklingWrapper

        if self.duckling is None:
            try:
                self.duckling = DucklingWrapper(language=language)  # languages in duckling are eg "de$core"
            except ValueError as e:
                raise Exception("Duckling error. {}".format(e))

    def process(self, text, entities):
        # type: (Text, List[Dict[Text, Any]]) -> Dict[Text, Any]

        extracted = []
        if self.duckling is not None:
            matches = self.duckling.parse(text)
            relevant_matches = [match for match in matches if match["dim"] in self.dimensions]
            for match in relevant_matches:
                entity = {"start": match["start"],
                          "end": match["end"],
                          "value": match["value"]["value"],
                          "entity": match["dim"]}

                extracted.append(entity)

        extracted = self.add_extractor_name(extracted)
        entities.extend(extracted)
        return {
            "entities": entities
        }

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        file_name = self.name+".json"
        full_name = os.path.join(model_dir, file_name)
        with io.open(full_name, 'w') as f:
            f.write(str(json.dumps({"dimensions": self.dimensions})))
        return {"ner_duckling_persisted": file_name}

    @classmethod
    def load(cls, model_dir, ner_duckling_persisted):
        # type: (Text) -> DucklingExtractor
        persisted = os.path.join(model_dir, ner_duckling_persisted)
        if os.path.isfile(persisted):
            with io.open(persisted, encoding='utf-8') as f:
                persisted_data = json.loads(f.read())
                return cls.create(persisted_data["dimensions"])
