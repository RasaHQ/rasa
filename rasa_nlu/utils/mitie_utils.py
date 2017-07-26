from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import typing
from builtins import str
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata

if typing.TYPE_CHECKING:
    import mitie


class MitieNLP(Component):
    name = "nlp_mitie"

    provides = ["mitie_feature_extractor"]

    def __init__(self, mitie_file, extractor=None):
        self.extractor = extractor
        self.mitie_file = mitie_file
        MitieNLP.ensure_proper_language_model(self.extractor)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    @classmethod
    def create(cls, config):
        import mitie
        return MitieNLP(config["mitie_file"], mitie.total_word_feature_extractor(config["mitie_file"]))

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        mitie_file = model_metadata.metadata.get("mitie_file", None)
        if mitie_file is not None:
            return cls.name + "-" + str(os.path.abspath(mitie_file))
        else:
            return None

    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {"mitie_feature_extractor": self.extractor}

    @staticmethod
    def ensure_proper_language_model(extractor):
        # type: (Optional[mitie.total_word_feature_extractor]) -> None

        if extractor is None:
            raise Exception("Failed to load MITIE feature extractor. Loading the model returned 'None'.")

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[MitieNLP], **Any) -> MitieNLP
        import mitie

        if cached_component:
            return cached_component

        mitie_file = model_metadata.get("mitie_file")
        return MitieNLP(mitie_file, mitie.total_word_feature_extractor(mitie_file))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "mitie_feature_extractor_fingerprint": self.extractor.fingerprint,
            "mitie_file": self.mitie_file
        }
