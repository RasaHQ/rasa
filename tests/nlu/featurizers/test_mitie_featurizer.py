import pytest
import typing
import numpy as np

from typing import Text, Dict, Any, Callable

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource

from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.featurizers.dense_featurizer.mitie_featurizer import (
    MitieFeaturizerGraphComponent,
)

if typing.TYPE_CHECKING:
    import mitie


@pytest.fixture
def resource() -> Resource:
    return Resource("MitieFeaturizerGraphComponent")


@pytest.fixture
def create(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    resource: Resource,
) -> Callable[[Dict[Text, Any]], MitieFeaturizerGraphComponent]:
    def inner(config: Dict[Text, Any]):
        return MitieFeaturizerGraphComponent.create(
            config={**MitieFeaturizerGraphComponent.get_default_config(), **config,},
            model_storage=default_model_storage,
            execution_context=default_execution_context,
            resource=resource,
        )

    return inner


def test_mitie_featurizer(
    create: Callable[[Dict[Text, Any]], MitieFeaturizerGraphComponent],
    mitie_feature_extractor: "mitie.total_word_feature_extractor",
):

    featurizer = create({"alias": "mitie_featurizer"})

    sentence = "Hey how are you today"
    message = Message(data={TEXT: sentence})
    MitieTokenizer().process(message)
    tokens = message.get(TOKENS_NAMES[TEXT])

    seq_vec, sen_vec = featurizer.features_for_tokens(tokens, mitie_feature_extractor)

    expected = np.array(
        [0.00000000e00, -5.12735510e00, 4.39929873e-01, -5.60760403e00, -8.26445103e00]
    )
    expected_cls = np.array([0.0, -4.4551446, 0.26073121, -1.46632245, -1.84205751])

    assert 6 == len(seq_vec) + len(sen_vec)
    assert np.allclose(seq_vec[0][:5], expected, atol=1e-5)
    assert np.allclose(sen_vec[-1][:5], expected_cls, atol=1e-5)


def test_mitie_featurizer_train(
    create: Callable[[Dict[Text, Any]], MitieFeaturizerGraphComponent],
    mitie_feature_extractor: "mitie.total_word_feature_extractor",
):

    featurizer = create({"alias": "mitie_featurizer"})

    sentence = "Hey how are you today"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    MitieTokenizer().train(TrainingData([message]))

    featurizer.train_process(
        TrainingData([message]), **{"mitie_feature_extractor": mitie_feature_extractor},
    )

    expected = np.array(
        [0.00000000e00, -5.12735510e00, 4.39929873e-01, -5.60760403e00, -8.26445103e00]
    )
    expected_cls = np.array([0.0, -4.4551446, 0.26073121, -1.46632245, -1.84205751])

    seq_vec, sen_vec = message.get_dense_features(TEXT, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert len(message.get(TOKENS_NAMES[TEXT])) == len(seq_vec)
    assert np.allclose(seq_vec[0][:5], expected, atol=1e-5)
    assert np.allclose(sen_vec[-1][:5], expected_cls, atol=1e-5)

    seq_vec, sen_vec = message.get_dense_features(RESPONSE, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert len(message.get(TOKENS_NAMES[RESPONSE])) == len(seq_vec)
    assert np.allclose(seq_vec[0][:5], expected, atol=1e-5)
    assert np.allclose(sen_vec[-1][:5], expected_cls, atol=1e-5)

    seq_vec, sen_vec = message.get_dense_features(INTENT, [])
    if seq_vec:
        seq_vec = seq_vec.features
    if sen_vec:
        sen_vec = sen_vec.features

    assert seq_vec is None
    assert sen_vec is None
