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
from rasa_nlu.model import Metadata


if typing.TYPE_CHECKING:
    import mitie


class MitieNLP(Component):
    name = "nlp_mitie"

    context_provides = {
        "pipeline_init": ["mitie_feature_extractor"],
    }

    def __init__(self, mitie_file, extractor=None):
        self.extractor = extractor
        self.mitie_file = mitie_file
        MitieNLP.ensure_proper_language_model(self.extractor)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    @classmethod
    def create(cls, mitie_file):
        import mitie
        extractor = mitie.total_word_feature_extractor(mitie_file)
        return MitieNLP(mitie_file, extractor)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        mitie_file = model_metadata.metadata.get("mitie_file", None)
        if mitie_file is not None:
            return cls.name + "-" + str(os.path.abspath(mitie_file))
        else:
            return None

    def pipeline_init(self, mitie_file):
        # type: (Text) -> Dict[Text, Any]

        return {"mitie_feature_extractor": self.extractor}

    @staticmethod
    def ensure_proper_language_model(extractor):
        # type: (Optional[mitie.total_word_feature_extractor]) -> None

        if extractor is None:
            raise Exception("Failed to load MITIE feature extractor. Loading the model returned 'None'.")

    @classmethod
    def load(cls, mitie_file):
        # type: (Text, Text) -> MitieNLP

        return cls.create(mitie_file)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "mitie_feature_extractor_fingerprint": self.extractor.fingerprint,
            "mitie_file": self.mitie_file
        }
