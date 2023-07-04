import numpy as np
import pytest
from typing import Text, Optional, List, Tuple, Dict, Any, Callable
from pathlib import Path
import os
from _pytest.monkeypatch import MonkeyPatch

from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
    NUMBER_OF_SUB_TOKENS,
)
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import (
    ConveRTFeaturizer,
    RESTRICTED_ACCESS_URL,
    ORIGINAL_TF_HUB_MODULE_URL,
)
from rasa.exceptions import RasaException
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.engine.storage.resource import Resource


@pytest.fixture
def create_or_load_convert_featurizer(
    default_model_storage: ModelStorage, default_execution_context: ExecutionContext
) -> Callable[[Dict[Text, Any], bool], ConveRTFeaturizer]:
    def inner(
        config: Dict[Text, Any], load: bool = False
    ) -> Callable[[Dict[Text, Any], bool], ConveRTFeaturizer]:
        if load:
            constructor = ConveRTFeaturizer.load
        else:
            constructor = ConveRTFeaturizer.create
        return constructor(
            config,
            model_storage=default_model_storage,
            execution_context=default_execution_context,
            resource=Resource("unused"),
        )

    return inner


@pytest.mark.skip_on_windows
def test_convert_featurizer_process(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    monkeypatch: MonkeyPatch,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    monkeypatch.setattr(
        ConveRTFeaturizer, "_validate_model_url", lambda _: RESTRICTED_ACCESS_URL
    )
    component_config = {
        FEATURIZER_CLASS_ALIAS: "alias",
        "model_url": RESTRICTED_ACCESS_URL,
    }
    featurizer = create_or_load_convert_featurizer(component_config)
    sentence = "Hey how are you today ?"
    message = Message.build(text=sentence)

    td = TrainingData([message])
    whitespace_tokenizer.process_training_data(td)
    tokens = featurizer.tokenize(message, attribute=TEXT)

    featurizer.process([message])

    expected = np.array([2.2636216, -0.26475656, -1.1358104, -0.49751878, -1.3946456])
    expected_cls = np.array(
        [1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353]
    )

    seq_vecs, sent_vecs = message.get_dense_features(TEXT, [])

    seq_vecs = seq_vecs.features
    sent_vecs = sent_vecs.features

    assert len(tokens) == len(seq_vecs)
    assert np.allclose(seq_vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(sent_vecs[-1][:5], expected_cls, atol=1e-5)


@pytest.mark.skip_on_windows
@pytest.mark.parametrize("load", [True, False])
def test_convert_featurizer_train(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    monkeypatch: MonkeyPatch,
    load: bool,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    monkeypatch.setattr(ConveRTFeaturizer, "_validate_model_url", lambda _: None)
    component_config = {
        FEATURIZER_CLASS_ALIAS: "alias",
        "model_url": RESTRICTED_ACCESS_URL,
    }
    featurizer = create_or_load_convert_featurizer(component_config, load=True)

    sentence = "Hey how are you today ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)

    td = TrainingData([message])
    whitespace_tokenizer.process_training_data(td)

    tokens = featurizer.tokenize(message, attribute=TEXT)

    message.set(TOKENS_NAMES[TEXT], tokens)
    message.set(TOKENS_NAMES[RESPONSE], tokens)

    featurizer.process_training_data(TrainingData([message]))

    expected = np.array([2.2636216, -0.26475656, -1.1358104, -0.49751878, -1.3946456])
    expected_cls = np.array(
        [1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353]
    )

    seq_vecs, sent_vecs = message.get_dense_features(TEXT, [])

    seq_vecs = seq_vecs.features
    sent_vecs = sent_vecs.features

    assert len(tokens) == len(seq_vecs)
    assert np.allclose(seq_vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(sent_vecs[-1][:5], expected_cls, atol=1e-5)

    seq_vecs, sent_vecs = message.get_dense_features(RESPONSE, [])

    seq_vecs = seq_vecs.features
    sent_vecs = sent_vecs.features

    assert len(tokens) == len(seq_vecs)
    assert np.allclose(seq_vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(sent_vecs[-1][:5], expected_cls, atol=1e-5)

    seq_vecs, sent_vecs = message.get_dense_features(INTENT, [])

    assert seq_vecs is None
    assert sent_vecs is None


@pytest.mark.skip_on_windows
@pytest.mark.parametrize(
    "sentence, expected_text",
    [
        ("hello", "hello"),
        ("you're", "you re"),
        ("r. n. b.", "r n b"),
        ("rock & roll", "rock & roll"),
        ("ńöñàśçií", "ńöñàśçií"),
    ],
)
def test_convert_featurizer_tokens_to_text(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    sentence: Text,
    expected_text: Text,
    monkeypatch: MonkeyPatch,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    monkeypatch.setattr(ConveRTFeaturizer, "_validate_model_url", lambda _: None)
    component_config = {
        FEATURIZER_CLASS_ALIAS: "alias",
        "model_url": RESTRICTED_ACCESS_URL,
    }
    featurizer = create_or_load_convert_featurizer(component_config)
    message = Message.build(text=sentence)
    td = TrainingData([message])
    whitespace_tokenizer.process_training_data(td)
    tokens = featurizer.tokenize(message, attribute=TEXT)

    actual_text = ConveRTFeaturizer._tokens_to_text([tokens])[0]

    assert expected_text == actual_text


@pytest.mark.skip_on_windows
@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "forecast for lunch",
            ["forecast", "for", "lunch"],
            [(0, 8), (9, 12), (13, 18)],
        ),
        ("hello", ["hello"], [(0, 5)]),
        ("you're", ["you", "re"], [(0, 3), (4, 6)]),
        ("r. n. b.", ["r", "n", "b"], [(0, 1), (3, 4), (6, 7)]),
        ("rock & roll", ["rock", "&", "roll"], [(0, 4), (5, 6), (7, 11)]),
        ("ńöñàśçií", ["ńöñàśçií"], [(0, 8)]),
    ],
)
def test_convert_featurizer_token_edge_cases(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    text: Text,
    expected_tokens: List[Text],
    expected_indices: List[Tuple[int, int]],
    monkeypatch: MonkeyPatch,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    monkeypatch.setattr(ConveRTFeaturizer, "_validate_model_url", lambda _: None)
    component_config = {
        FEATURIZER_CLASS_ALIAS: "alias",
        "model_url": RESTRICTED_ACCESS_URL,
    }
    featurizer = create_or_load_convert_featurizer(component_config)
    message = Message.build(text=text)
    td = TrainingData([message])
    whitespace_tokenizer.process_training_data(td)
    tokens = featurizer.tokenize(message, attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.skip_on_windows
@pytest.mark.parametrize(
    "text, expected_number_of_sub_tokens",
    [("Aarhus is a city", [2, 1, 1, 1]), ("sentence embeddings", [1, 3])],
)
def test_convert_featurizer_number_of_sub_tokens(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    text: Text,
    expected_number_of_sub_tokens: List[int],
    monkeypatch: MonkeyPatch,
    whitespace_tokenizer: WhitespaceTokenizer,
):
    monkeypatch.setattr(ConveRTFeaturizer, "_validate_model_url", lambda _: None)
    component_config = {
        FEATURIZER_CLASS_ALIAS: "alias",
        "model_url": RESTRICTED_ACCESS_URL,
    }
    featurizer = create_or_load_convert_featurizer(component_config)

    message = Message.build(text=text)
    td = TrainingData([message])
    whitespace_tokenizer.process_training_data(td)

    tokens = featurizer.tokenize(message, attribute=TEXT)

    assert [
        t.get(NUMBER_OF_SUB_TOKENS) for t in tokens
    ] == expected_number_of_sub_tokens


@pytest.mark.skip_on_windows
@pytest.mark.parametrize(
    "model_url, exception_phrase",
    [
        (ORIGINAL_TF_HUB_MODULE_URL, "which does not contain the model any longer"),
        (
            RESTRICTED_ACCESS_URL,
            "which is strictly reserved for pytests of Rasa Open Source only",
        ),
        (None, "'model_url' was not specified in the configuration"),
        ("", "'model_url' was not specified in the configuration"),
    ],
)
def test_raise_invalid_urls(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    model_url: Optional[Text],
    exception_phrase: Text,
):
    component_config = {FEATURIZER_CLASS_ALIAS: "alias", "model_url": model_url}
    with pytest.raises(RasaException) as excinfo:
        _ = create_or_load_convert_featurizer(component_config)

    assert exception_phrase in str(excinfo.value)


@pytest.mark.skip_on_windows
def test_raise_wrong_model_directory(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    tmp_path: Path,
):
    component_config = {FEATURIZER_CLASS_ALIAS: "alias", "model_url": str(tmp_path)}

    with pytest.raises(RasaException) as excinfo:
        _ = create_or_load_convert_featurizer(component_config)

    assert "Re-check the files inside the directory" in str(excinfo.value)


@pytest.mark.skip_on_windows
def test_raise_wrong_model_file(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer],
    tmp_path: Path,
):
    # create a dummy file
    temp_file = os.path.join(tmp_path, "saved_model.pb")
    f = open(temp_file, "wb")
    f.close()
    component_config = {FEATURIZER_CLASS_ALIAS: "alias", "model_url": temp_file}

    with pytest.raises(RasaException) as excinfo:
        _ = create_or_load_convert_featurizer(component_config)

    assert "set to the path of a file which is invalid" in str(excinfo.value)


@pytest.mark.skip_on_windows
def test_raise_invalid_path(
    create_or_load_convert_featurizer: Callable[[Dict[Text, Any]], ConveRTFeaturizer]
):
    component_config = {FEATURIZER_CLASS_ALIAS: "alias", "model_url": "saved_model.pb"}

    with pytest.raises(RasaException) as excinfo:
        _ = create_or_load_convert_featurizer(component_config)

    assert "neither a valid remote URL nor a local directory" in str(excinfo.value)
