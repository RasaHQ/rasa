import numpy as np

from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.constants import CLS_TOKEN


def test_mitie_featurizer(mitie_feature_extractor, default_config):
    from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer

    component_config = {"name": "MitieFeaturizer", "return_sequence": True}
    featurizer = MitieFeaturizer.create(component_config, RasaNLUModelConfig())

    sentence = f"Hey how are you today {CLS_TOKEN}"

    tokens = MitieTokenizer().tokenize(sentence)

    vecs = featurizer.features_for_tokens(tokens, mitie_feature_extractor)[0]

    expected = np.array(
        [0.00000000e00, -5.12735510e00, 4.39929873e-01, -5.60760403e00, -8.26445103e00]
    )
    assert np.allclose(vecs[:5], expected, atol=1e-5)


def test_mitie_featurizer_no_sequence(mitie_feature_extractor, default_config):
    from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer

    component_config = {"name": "MitieFeaturizer", "return_sequence": False}
    featurizer = MitieFeaturizer.create(component_config, RasaNLUModelConfig())

    sentence = f"Hey how are you today {CLS_TOKEN}"
    tokens = MitieTokenizer().tokenize(sentence)

    vecs = featurizer.features_for_tokens(tokens, mitie_feature_extractor)[0]

    expected = np.array([0.0, -4.4551446, 0.26073121, -1.46632245, -1.84205751])
    assert np.allclose(vecs[:5], expected, atol=1e-5)
