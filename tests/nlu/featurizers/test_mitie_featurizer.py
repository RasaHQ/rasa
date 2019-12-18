import numpy as np

from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import MitieFeaturizer


def test_mitie_featurizer(mitie_feature_extractor):

    featurizer = MitieFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "Hey how are you today"
    tokens = MitieTokenizer().tokenize(sentence)

    vecs = featurizer.features_for_tokens(tokens, mitie_feature_extractor)

    expected = np.array(
        [0.00000000e00, -5.12735510e00, 4.39929873e-01, -5.60760403e00, -8.26445103e00]
    )
    expected_cls = np.array([0.0, -4.4551446, 0.26073121, -1.46632245, -1.84205751])

    assert 6 == len(vecs)
    assert np.allclose(vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(vecs[-1][:5], expected_cls, atol=1e-5)
