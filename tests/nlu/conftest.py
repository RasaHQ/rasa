from typing import Text

import pytest

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import ComponentBuilder
from rasa.utils.tensorflow.constants import EPOCHS
from tests.nlu.utilities import write_file_config

DEFAULT_DATA_PATH = "data/examples/rasa/demo-rasa.json"


@pytest.fixture(scope="session")
def component_builder():
    return ComponentBuilder()


@pytest.fixture(scope="session")
def spacy_nlp(component_builder, default_config):
    spacy_nlp_config = {"name": "SpacyNLP"}
    return component_builder.create_component(spacy_nlp_config, default_config).nlp


@pytest.fixture(scope="session")
def spacy_nlp_component(component_builder, default_config):
    spacy_nlp_config = {"name": "SpacyNLP"}
    return component_builder.create_component(spacy_nlp_config, default_config)


@pytest.fixture(scope="session")
def mitie_feature_extractor(
    component_builder: ComponentBuilder, default_config: RasaNLUModelConfig
):
    mitie_nlp_config = {"name": "MitieNLP"}
    return component_builder.create_component(
        mitie_nlp_config, default_config
    ).extractor


@pytest.fixture(scope="session")
def default_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig({"language": "en", "pipeline": []})


@pytest.fixture(scope="session")
def config_path() -> Text:
    return write_file_config(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CRFEntityExtractor", EPOCHS: 2},
                {"name": "CountVectorsFeaturizer"},
                {"name": "EmbeddingIntentClassifier", EPOCHS: 2},
            ],
        }
    ).name


@pytest.fixture(scope="session")
def pretrained_embeddings_spacy_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "SpacyNLP"},
                {"name": "SpacyTokenizer"},
                {"name": "SpacyFeaturizer"},
                {"name": "RegexFeaturizer"},
                {"name": "CRFEntityExtractor", EPOCHS: 3},
                {"name": "EntitySynonymMapper"},
                {"name": "SklearnIntentClassifier"},
            ],
        }
    )


@pytest.fixture(scope="session")
def supervised_embeddings_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "RegexFeaturizer"},
                {"name": "CRFEntityExtractor", EPOCHS: 3},
                {"name": "EntitySynonymMapper"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "CountVectorsFeaturizer",
                    "analyzer": "char_wb",
                    "min_ngram": 1,
                    "max_ngram": 4,
                },
                {"name": "EmbeddingIntentClassifier", EPOCHS: 3},
            ],
        }
    )


@pytest.fixture(scope="session")
def pretrained_embeddings_convert_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig(
        {
            "language": "en",
            "pipeline": [
                {"name": "ConveRTTokenizer"},
                {"name": "ConveRTFeaturizer"},
                {"name": "EmbeddingIntentClassifier", EPOCHS: 3},
            ],
        }
    )
