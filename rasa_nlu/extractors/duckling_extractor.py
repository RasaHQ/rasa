from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import io
import json
import logging
import typing
import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from builtins import str

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from inspect import getmembers

from rasa_nlu.training_data import Message
from rasa_nlu.utils import write_json_to_file

if typing.TYPE_CHECKING:
    from duckling import DucklingWrapper


def extract_value(match):
    if match["value"].get("type") == "interval":
        value = {"to": match["value"].get("to", {}).get("value"),
                 "from": match["value"].get("from", {}).get("value")}
    else:
        value = match["value"].get("value")

    return value


class DucklingExtractor(EntityExtractor):
    """Adds entity normalization by analyzing found entities and
    transforming them into regular formats."""

    name = "ner_duckling"

    provides = ["entities"]

    @staticmethod
    def available_dimensions():
        from duckling.dim import Dim
        return [m[1]
                for m in getmembers(Dim)
                if not m[0].startswith("__") and not m[0].endswith("__")]

    def __init__(self, duckling, dimensions=None):
        # type: (DucklingWrapper, Optional[List[Text]]) -> None

        super(DucklingExtractor, self).__init__()
        self.duckling = duckling

        if dimensions is not None:
            self.dimensions = dimensions
        else:
            self.dimensions = self.available_dimensions()

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["duckling"]

    @classmethod
    def create_duckling_wrapper(cls, language):
        from duckling import DucklingWrapper

        try:
            # languages in duckling are eg "de$core"
            return DucklingWrapper(language=language)
        except ValueError as e:  # pragma: no cover
            raise Exception("Duckling error. {}".format(e))

    @classmethod
    def create(cls, config):
        # type: (RasaNLUConfig) -> DucklingExtractor

        dims = config["duckling_dimensions"]
        if dims:
            unknown_dimensions = [dim
                                  for dim in dims
                                  if dim not in cls.available_dimensions()]
            if len(unknown_dimensions) > 0:
                raise ValueError(
                        "Invalid duckling dimension. Got '{}'. Allowed: {}"
                        "".format(", ".join(unknown_dimensions),
                                  ", ".join(cls.available_dimensions())))

        wrapper = cls.create_duckling_wrapper(config["language"])
        return DucklingExtractor(wrapper, dims)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        return cls.name + "-" + model_metadata.language

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        if self.duckling is None:
            return

        extracted = []

        current_time = datetime.datetime.utcnow()
        ref_time = current_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')
        if message.time is not None:
            # check if time given is valid
            try:
                ref_time = datetime.datetime \
                    .utcfromtimestamp(int(message.time) / 1000.0) \
                    .strftime('%Y-%m-%dT%H:%M:%S+00:00')
                logging.debug("Passing reference time {} "
                              "to duckling".format(ref_time))
            except Exception as e:
                logging.warning("Could not parse timestamp {}. Instead "
                                "current UTC time {} will be passed to "
                                "duckling. Error: {}"
                                "".format(message.time, ref_time, e))

        matches = self.duckling.parse(message.text, reference_time=ref_time)
        relevant_matches = [match
                            for match in matches
                            if match["dim"] in self.dimensions]

        for match in relevant_matches:
            value = extract_value(match)
            entity = {"start": match["start"],
                      "end": match["end"],
                      "text": match["text"],
                      "value": value,
                      "additional_info": match["value"],
                      "entity": match["dim"]}

            extracted.append(entity)

        extracted = self.add_extractor_name(extracted)
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        file_name = self.name + ".json"
        full_name = os.path.join(model_dir, file_name)
        write_json_to_file(full_name, {"dimensions": self.dimensions})
        return {"ner_duckling_persisted": file_name}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type:Optional[DucklingExtractor]
             **kwargs  # type: **Any
             ):
        # type: (...) -> DucklingExtractor

        persisted = os.path.join(model_dir,
                                 model_metadata.get("ner_duckling_persisted"))
        if cached_component:
            duckling = cached_component.duckling
        else:
            language = model_metadata.get("language")
            duckling = cls.create_duckling_wrapper(language)

        if os.path.isfile(persisted):
            with io.open(persisted, encoding='utf-8') as f:
                persisted_data = json.loads(f.read())
                return DucklingExtractor(duckling, persisted_data["dimensions"])
        return DucklingExtractor(duckling)
