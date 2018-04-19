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
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata

if typing.TYPE_CHECKING:
    import mitie


class MitieNLP(Component):
    name = "nlp_mitie"

    provides = ["mitie_feature_extractor", "mitie_file"]

    defaults = {
        # name of the language model to load - this contains
        # the MITIE feature extractor
        "model": os.path.join("data", "total_word_feature_extractor.dat"),
    }

    def __init__(self,
                 component_config=None,  # type: Dict[Text, Any]
                 extractor=None
                 ):
        # type: (...) -> None
        """Construct a new language model from the MITIE framework."""

        super(MitieNLP, self).__init__(component_config)

        self.extractor = extractor

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie"]

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> MitieNLP
        import mitie

        component_conf = cfg.for_component(cls.name, cls.defaults)
        model_file = component_conf.get("model")
        if not model_file:
            raise Exception("The MITIE component 'nlp_mitie' needs "
                            "the configuration value for 'model'."
                            "Please take a look at the "
                            "documentation in the pipeline section "
                            "to get more info about this "
                            "parameter.")
        extractor = mitie.total_word_feature_extractor(model_file)
        cls.ensure_proper_language_model(extractor)

        return MitieNLP(component_conf, extractor)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Optional[Text]

        component_meta = model_metadata.for_component(cls.name)

        mitie_file = component_meta.get("model", None)
        if mitie_file is not None:
            return cls.name + "-" + str(os.path.abspath(mitie_file))
        else:
            return None

    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {"mitie_feature_extractor": self.extractor,
                "mitie_file": self.component_config.get("model")}

    @staticmethod
    def ensure_proper_language_model(extractor):
        # type: (Optional[mitie.total_word_feature_extractor]) -> None

        if extractor is None:
            raise Exception("Failed to load MITIE feature extractor. "
                            "Loading the model returned 'None'.")

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[MitieNLP]
             **kwargs  # type: **Any
             ):
        # type: (...) -> MitieNLP
        import mitie

        if cached_component:
            return cached_component

        component_meta = model_metadata.for_component(cls.name)
        mitie_file = component_meta.get("model")
        return cls(component_meta,
                   mitie.total_word_feature_extractor(mitie_file))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]

        return {
            "mitie_feature_extractor_fingerprint": self.extractor.fingerprint,
            "model": self.component_config.get("model")
        }
