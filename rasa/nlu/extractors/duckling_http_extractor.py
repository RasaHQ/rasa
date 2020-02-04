import time
import json
import logging
import os
import requests
from typing import Any, List, Optional, Text, Dict

from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu.constants import ENTITIES_ATTRIBUTE
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message
from rasa.utils.common import raise_warning

logger = logging.getLogger(__name__)


def extract_value(match: Dict[Text, Any]) -> Dict[Text, Any]:
    if match["value"].get("type") == "interval":
        value = {
            "to": match["value"].get("to", {}).get("value"),
            "from": match["value"].get("from", {}).get("value"),
        }
    else:
        value = match["value"].get("value")

    return value


def convert_duckling_format_to_rasa(
    matches: List[Dict[Text, Any]]
) -> List[Dict[Text, Any]]:
    extracted = []

    for match in matches:
        value = extract_value(match)
        entity = {
            "start": match["start"],
            "end": match["end"],
            "text": match.get("body", match.get("text", None)),
            "value": value,
            "confidence": 1.0,
            "additional_info": match["value"],
            "entity": match["dim"],
        }

        extracted.append(entity)

    return extracted


class DucklingHTTPExtractor(EntityExtractor):
    """Searches for structured entites, e.g. dates, using a duckling server."""

    provides = [ENTITIES_ATTRIBUTE]

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
        "timezone": None,
        # Timeout for receiving response from http url of the running duckling server
        # if not set the default timeout of duckling http url is set to 3 seconds.
        "timeout": 3,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        language: Optional[Text] = None,
    ) -> None:

        super().__init__(component_config)
        self.language = language

    @classmethod
    def create(
        cls, component_config: Dict[Text, Any], config: RasaNLUModelConfig
    ) -> "DucklingHTTPExtractor":

        return cls(component_config, config.language)

    def _locale(self) -> Optional[Text]:
        if not self.component_config.get("locale"):
            # this is king of a quick fix to generate a proper locale
            # works most of the time
            language = self.language or ""
            locale_fix = "{}_{}".format(language, language.upper())
            self.component_config["locale"] = locale_fix
        return self.component_config.get("locale")

    def _url(self) -> Optional[Text]:
        """Return url of the duckling service. Environment var will override."""
        if os.environ.get("RASA_DUCKLING_HTTP_URL"):
            return os.environ["RASA_DUCKLING_HTTP_URL"]

        return self.component_config.get("url")

    def _payload(self, text: Text, reference_time: int) -> Dict[Text, Any]:
        dimensions = self.component_config["dimensions"]
        return {
            "text": text,
            "locale": self._locale(),
            "tz": self.component_config.get("timezone"),
            "dims": json.dumps(dimensions),
            "reftime": reference_time,
        }

    def _duckling_parse(self, text: Text, reference_time: int) -> List[Dict[Text, Any]]:
        """Sends the request to the duckling server and parses the result."""

        try:
            payload = self._payload(text, reference_time)
            headers = {
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }
            response = requests.post(
                self._url() + "/parse",
                data=payload,
                headers=headers,
                timeout=self.component_config.get("timeout"),
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    "Failed to get a proper response from remote "
                    "duckling. Status Code: {}. Response: {}"
                    "".format(response.status_code, response.text)
                )
                return []
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ) as e:
            logger.error(
                "Failed to connect to duckling http server. Make sure "
                "the duckling server is running/healthy/not stale and the proper host "
                "and port are set in the configuration. More "
                "information on how to run the server can be found on "
                "github: "
                "https://github.com/facebook/duckling#quickstart "
                "Error: {}".format(e)
            )
            return []

    @staticmethod
    def _reference_time_from_message(message: Message) -> int:
        if message.time is not None:
            try:
                return int(message.time) * 1000
            except ValueError as e:
                logging.warning(
                    "Could not parse timestamp {}. Instead "
                    "current UTC time will be passed to "
                    "duckling. Error: {}".format(message.time, e)
                )
        # fallbacks to current time, multiplied by 1000 because duckling
        # requires the reftime in miliseconds
        return int(time.time()) * 1000

    def process(self, message: Message, **kwargs: Any) -> None:

        if self._url() is not None:
            reference_time = self._reference_time_from_message(message)
            matches = self._duckling_parse(message.text, reference_time)
            all_extracted = convert_duckling_format_to_rasa(matches)
            dimensions = self.component_config["dimensions"]
            extracted = DucklingHTTPExtractor.filter_irrelevant_entities(
                all_extracted, dimensions
            )
        else:
            extracted = []
            raise_warning(
                "Duckling HTTP component in pipeline, but no "
                "`url` configuration in the config "
                "file nor is `RASA_DUCKLING_HTTP_URL` "
                "set as an environment variable. No entities will be extracted!",
                docs=DOCS_URL_COMPONENTS + "#ducklinghttpextractor",
            )

        extracted = self.add_extractor_name(extracted)
        message.set(
            ENTITIES_ATTRIBUTE,
            message.get(ENTITIES_ATTRIBUTE, []) + extracted,
            add_to_output=True,
        )

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["DucklingHTTPExtractor"] = None,
        **kwargs: Any,
    ) -> "DucklingHTTPExtractor":

        language = model_metadata.get("language") if model_metadata else None
        return cls(meta, language)
