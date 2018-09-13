from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time

import requests
import simplejson
from typing import Any
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.extractors import EntityExtractor
from rasa_nlu.extractors.duckling_extractor import (
    filter_irrelevant_matches, convert_duckling_format_to_rasa)
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

logger = logging.getLogger(__name__)


class DucklingHTTPExtractor(EntityExtractor):
    """Searches for structured entites, e.g. dates, using a duckling server."""

    name = "ner_duckling_http"

    provides = ["entities"]

    defaults = {
        # by default all dimensions recognized by duckling are returned
        # dimensions can be configured to contain an array of strings
        # with the names of the dimensions to filter for
        "dimensions": None,

        # http url of the running duckling server
        "url": None,

        # locale - if not set, we will use the language of the model
        "locale": None,

        # timezone like Europe/Berlin
        # if not set the default timezone of Duckling is going to be used
        "timezone": None
    }

    def __init__(self, component_config=None, language=None):
        # type: (Text, Optional[List[Text]]) -> None

        super(DucklingHTTPExtractor, self).__init__(component_config)
        self.language = language

    @classmethod
    def create(cls, config):
        # type: (RasaNLUModelConfig) -> DucklingHTTPExtractor

        return cls(config.for_component(cls.name,
                                        cls.defaults),
                   config.language)

    def _locale(self):
        if not self.component_config.get("locale"):
            # this is king of a quick fix to generate a proper locale
            # works most of the time
            locale_fix = "{}_{}".format(self.language, self.language.upper())
            self.component_config["locale"] = locale_fix
        return self.component_config.get("locale")

    def _url(self):
        """Return url of the duckling service. Environment var will override."""
        if os.environ.get("RASA_DUCKLING_HTTP_URL"):
            return os.environ["RASA_DUCKLING_HTTP_URL"]

        return self.component_config.get("url")

    def _payload(self, text, reference_time):
        return {
            "text": text,
            "locale": self._locale(),
            "tz": self.component_config.get("timezone"),
            "reftime": reference_time
        }

    def _duckling_parse(self, text, reference_time):
        """Sends the request to the duckling server and parses the result."""

        try:
            payload = self._payload(text, reference_time)
            headers = {"Content-Type": "application/x-www-form-urlencoded; "
                                       "charset=UTF-8"}
            response = requests.post(self._url() + "/parse",
                                     data=payload,
                                     headers=headers)
            if response.status_code == 200:
                return simplejson.loads(response.text)
            else:
                logger.error("Failed to get a proper response from remote "
                             "duckling. Status Code: {}. Response: {}"
                             "".format(response.status_code, response.text))
                return []
        except requests.exceptions.ConnectionError as e:
            logger.error("Failed to connect to duckling http server. Make sure "
                         "the duckling server is running and the proper host "
                         "and port are set in the configuration. More "
                         "information on how to run the server can be found on "
                         "github: "
                         "https://github.com/facebook/duckling#quickstart "
                         "Error: {}".format(e))
            return []

    @staticmethod
    def _reference_time_from_message(message):
        if message.time is not None:
            try:
                return int(message.time) * 1000
            except ValueError as e:
                logging.warning("Could not parse timestamp {}. Instead "
                                "current UTC time will be passed to "
                                "duckling. Error: {}".format(message.time, e))
        # fallbacks to current time, multiplied by 1000 because duckling
        # requires the reftime in miliseconds
        return int(time.time()) * 1000

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        if self._url() is not None:
            reference_time = self._reference_time_from_message(message)
            matches = self._duckling_parse(message.text, reference_time)
            dimensions = self.component_config["dimensions"]
            relevant_matches = filter_irrelevant_matches(matches, dimensions)
            extracted = convert_duckling_format_to_rasa(relevant_matches)
        else:
            extracted = []
            logger.warning("Duckling HTTP component in pipeline, but no "
                           "`url` configuration in the config "
                           "file nor is `RASA_DUCKLING_HTTP_URL` "
                           "set as an environment variable.")

        extracted = self.add_extractor_name(extracted)
        message.set("entities",
                    message.get("entities", []) + extracted,
                    add_to_output=True)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[DucklingHTTPExtractor]
             **kwargs  # type: **Any
             ):
        # type: (...) -> DucklingHTTPExtractor

        component_config = model_metadata.for_component(cls.name)
        return cls(component_config, model_metadata.get("language"))
