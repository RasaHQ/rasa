import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.utils.tensorflow.constants import EPOCHS, RANDOM_SEED


@pytest.fixture()
def pretrained_embeddings_spacy_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "SpacyNLP", "model": "en_core_web_md"},
                {"name": "SpacyTokenizer"},
                {"name": "SpacyFeaturizer"},
                {"name": "RegexFeaturizer"},
                {"name": "CRFEntityExtractor", EPOCHS: 1, RANDOM_SEED: 42},
                {"name": "EntitySynonymMapper"},
                {"name": "SklearnIntentClassifier"},
            ],
        }
    )


@pytest.fixture()
def supervised_embeddings_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "RegexFeaturizer"},
                {"name": "CRFEntityExtractor", EPOCHS: 1, RANDOM_SEED: 42},
                {"name": "EntitySynonymMapper"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "CountVectorsFeaturizer",
                    "analyzer": "char_wb",
                    "min_ngram": 1,
                    "max_ngram": 4,
                },
                {"name": "DIETClassifier", EPOCHS: 1, RANDOM_SEED: 42},
            ],
        }
    )


@pytest.fixture()
def pretrained_embeddings_convert_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "ConveRTFeaturizer"},
                {"name": "DIETClassifier", EPOCHS: 1, RANDOM_SEED: 42},
            ],
        }
    )
