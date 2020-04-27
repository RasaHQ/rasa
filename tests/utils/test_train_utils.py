import numpy as np

import rasa.utils.train_utils as train_utils
from rasa.nlu.constants import TEXT, LANGUAGE_MODEL_DOCS, TOKENS, SEQUENCE_FEATURES
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.tokenizers.lm_tokenizer import LanguageModelTokenizer
from rasa.nlu.training_data import Message
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP


def test_align_token_features_convert():
    tokens = ConveRTTokenizer().tokenize(Message("In Aarhus and Ahaus"), attribute=TEXT)
    x = sum(t.get("number_of_sub_words") for t in tokens)
    token_features = np.random.rand(1, x, 64)

    actual_features = train_utils.align_token_features([tokens], token_features)

    assert np.all(actual_features[0][0] == token_features[0][0])
    assert np.all(actual_features[0][1] == np.mean(token_features[0][1:3], axis=0))
    assert np.all(actual_features[0][2] == token_features[0][3])
    assert np.all(actual_features[0][3] == np.mean(token_features[0][4:6], axis=0))


def test_align_token_features_lm():
    transformers_nlp = HFTransformersNLP({"model_name": "bert"})

    message = Message("word embeddings")
    transformers_nlp.process(message)

    doc = message.get(LANGUAGE_MODEL_DOCS[TEXT])

    tokens = doc[TOKENS]
    sequence_features = doc[SEQUENCE_FEATURES]

    assert len(tokens) == len(sequence_features)
