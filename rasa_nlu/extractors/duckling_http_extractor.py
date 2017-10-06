from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import os

import simplejson
from builtins import str
from typing import Any, Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message


class DucklingHTTPExtractor(EntityExtractor):
    """Adds entity normalization by analyzing found entities and transforming
    them into regular formats."""

    name = "ner_duckling_http"

    provides = ["entities"]

    def __init__(self, duckling_url, language, dimensions=None):
        # type: (Text, Optional[List[Text]]) -> None

        self.dimensions = dimensions if dimensions is not None else []
        self.duckling_url = duckling_url
        self.language = language

    @classmethod
    def create(cls, config):
        # type: (RasaNLUConfig) -> DucklingHTTPExtractor

        return DucklingHTTPExtractor(config["duckling_http_url"],
                                     config["language"],
                                     config["duckling_dimensions"])

    def _duckling_parse(self, text):
        import requests

        response = requests.post(self.duckling_url + "/parse",
                                 data={"text": text, "lang": self.language})

        return simplejson.loads(response.text)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        extracted = []
        if self.duckling_url is not None:

            matches = self._duckling_parse(message.text)
            relevant_matches = [match
                                for match in matches
                                if match["dim"] in self.dimensions]
            for match in relevant_matches:
                entity = {"start": match["start"],
                          "end": match["end"],
                          "text": match["body"],
                          "value": match["value"]["value"],
                          "additional_info": match["value"],
                          "entity": match["dim"]}

                extracted.append(entity)

        extracted = self.add_extractor_name(extracted)
        message.set("entities", message.get("entities", []) + extracted,
                    add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        file_name = self.name+".json"
        full_name = os.path.join(model_dir, file_name)
        with io.open(full_name, 'w') as f:
            f.write(str(simplejson.dumps({"dimensions": self.dimensions})))
        return {self.name: file_name}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[DucklingHTTPExtractor], **Any) -> DucklingHTTPExtractor

        persisted = os.path.join(model_dir, model_metadata.get(cls.name))
        config = kwargs.get("config", {})

        dimensions = None
        if os.path.isfile(persisted):
            with io.open(persisted, encoding='utf-8') as f:
                persisted_data = simplejson.loads(f.read())
                dimensions = persisted_data["dimensions"]
        return DucklingHTTPExtractor(config.get("duckling_http_url"),
                                     model_metadata.get("language"),
                                     dimensions)
