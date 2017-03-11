from mitie import total_word_feature_extractor
from rasa_nlu.components import Component


MITIE_BACKEND_NAME = "mitie"

MITIE_SKLEARN_BACKEND_NAME = "mitie_sklearn"


class MitieNLP(Component):

    name = "init_mitie"

    context_provides = ["mitie_feature_extractor"]

    def __init__(self, extractor=None):
        self.extractor = extractor

    @classmethod
    def cache_key(cls, model_metadata):
        return cls.name + "-" + str(model_metadata.metadata.get("mitie_feature_extractor_fingerprint"))

    def pipeline_init(self, mitie_file):
        if self.extractor is None:
            self.extractor = total_word_feature_extractor(mitie_file)
        MitieNLP.ensure_proper_language_model(self.extractor)
        return {"mitie_feature_extractor": self.extractor}

    @staticmethod
    def ensure_proper_language_model(extractor):
        if extractor is None:
            raise Exception("Failed to load MITIE feature extractor. Loading the model returned 'None'.")

    def persist(self, model_dir):
        return {
            "mitie_feature_extractor_fingerprint": self.extractor.fingerprint
        }
