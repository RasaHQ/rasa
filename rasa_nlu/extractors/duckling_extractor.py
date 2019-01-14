from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import logging
from inspect import getmembers

import typing
from typing import Any, Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

if typing.TYPE_CHECKING:
    from duckling import DucklingWrapper


def extract_value(match):
    if match["value"].get("type") == "interval":
        value = {"to": match["value"].get("to", {}).get("value"),
                 "from": match["value"].get("from", {}).get("value")}
    else:
        value = match["value"].get("value")

    return value


def filter_irrelevant_matches(matches, requested_dimensions):
    """Only return dimensions the user configured"""

    if requested_dimensions:
        return [match
                for match in matches
                if match["dim"] in requested_dimensions]
    else:
        return matches


def convert_duckling_format_to_rasa(matches):
    extracted = []

    for match in matches:
        value = extract_value(match)
        entity = {"start": match["start"],
                  "end": match["end"],
                  "text": match.get("body", match.get("text", None)),
                  "value": value,
                  "confidence": 1.0,
                  "additional_info": match["value"],
                  "entity": match["dim"]}

        extracted.append(entity)

    return extracted


def current_datetime_str():
    current_time = datetime.datetime.utcnow()
    return current_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')


class DucklingExtractor(EntityExtractor):
    """Adds entity normalization by analyzing found entities and
    transforming them into regular formats."""

    name = "ner_duckling"

    provides = ["entities"]

    defaults = {
        # by default all dimensions recognized by duckling are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None
    }

    @staticmethod
    def available_dimensions():
        from duckling.dim import Dim
        return [m[1]
                for m in getmembers(Dim)
                if not m[0].startswith("__") and not m[0].endswith("__")]

    def __init__(self, component_config=None, duckling=None):
        # type: (Dict[Text, Any], DucklingWrapper) -> None

        super(DucklingExtractor, self).__init__(component_config)
        self.duckling = duckling

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
        # type: (RasaNLUModelConfig) -> DucklingExtractor

        component_config = config.for_component(cls.name, cls.defaults)
        dims = component_config.get("dimensions")
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
        return DucklingExtractor(component_config, wrapper)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        return None

    @staticmethod
    def reference_time_from_message(message):
        # fallback to current time by default
        ref_time = current_datetime_str()

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
        return ref_time

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        if self.duckling is None:
            return

        ref_time = self.reference_time_from_message(message)

        try:
            matches = self.duckling.parse(message.text, reference_time=ref_time)
        except Exception as e:
            logging.warn("Invalid Duckling parse. Error {e}", e)
            matches = []

        dimensions = self.component_config["dimensions"]
        relevant_matches = filter_irrelevant_matches(matches, dimensions)

        extracted = convert_duckling_format_to_rasa(relevant_matches)

        extracted = self.add_extractor_name(extracted)

        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[DucklingExtractor]
             **kwargs  # type: **Any
             ):
        # type: (...) -> DucklingExtractor

        if cached_component:
            duckling = cached_component.duckling
        else:
            language = model_metadata.get("language")
            duckling = cls.create_duckling_wrapper(language)

        component_config = model_metadata.for_component(cls.name)
        return cls(component_config, duckling)
