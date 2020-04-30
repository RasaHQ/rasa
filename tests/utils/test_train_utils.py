import numpy as np

import rasa.utils.train_utils as train_utils
from rasa.nlu.constants import TEXT, NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.training_data import Message


def test_align_token_features_convert():
    tokens = ConveRTTokenizer().tokenize(Message("In Aarhus and Ahaus"), attribute=TEXT)
    seq_dim = sum(t.get(NUMBER_OF_SUB_TOKENS) for t in tokens)
    token_features = np.random.rand(1, seq_dim, 64)

    actual_features = train_utils.align_token_features([tokens], token_features)

    assert np.all(actual_features[0][0] == token_features[0][0])
    # Aarhus is split into 'aar' and 'hus'
    assert np.all(actual_features[0][1] == np.mean(token_features[0][1:3], axis=0))
    assert np.all(actual_features[0][2] == token_features[0][3])
    # Ahaus is split into 'aha' and 'us'
    assert np.all(actual_features[0][3] == np.mean(token_features[0][4:6], axis=0))
