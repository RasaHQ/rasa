import numpy as np

from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.config import RasaNLUModelConfig


def test_mitie_featurizer(mitie_feature_extractor, default_config):
    from nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer

    mitie_component_config = {"name": "MitieFeaturizer"}
    ftr = MitieFeaturizer.create(mitie_component_config, RasaNLUModelConfig())
    sentence = "Hey how are you today"
    mitie_component_config = {"name": "MitieTokenizer", "use_cls_token": False}
    tokens = MitieTokenizer(mitie_component_config).tokenize(sentence)
    vecs = ftr.features_for_tokens(tokens, mitie_feature_extractor)
    expected = np.array([0.0, -4.4551446, 0.26073121, -1.46632245, -1.84205751])
    assert np.allclose(vecs[:5], expected, atol=1e-5)
