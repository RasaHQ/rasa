import pytest
from typing import Text
import os
import tempfile

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES, NUMBER_OF_SUB_TOKENS
from rasa.shared.nlu.constants import TEXT, INTENT
from rasa.nlu.tokenizers.convert_tokenizer import (
    ConveRTTokenizer,
    RESTRICTED_ACCESS_URL,
    ORIGINAL_TF_HUB_MODULE_URL,
)
from rasa.exceptions import RasaException
import rasa.utils.io as io


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
def test_convert_tokenizer_edge_cases(text, expected_tokens, expected_indices):

    component_config = {"name": "ConveRTTokenizer", "model_url": RESTRICTED_ACCESS_URL}
    tokenizer = ConveRTTokenizer(component_config, ignore_exceptions=True)

    tokens = tokenizer.tokenize(Message(data={TEXT: text}), attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_tokens",
    [
        ("Forecast_for_LUNCH", ["Forecast_for_LUNCH"]),
        ("Forecast for LUNCH", ["Forecast for LUNCH"]),
    ],
)
def test_custom_intent_symbol(text, expected_tokens):

    component_config = {
        "name": "ConveRTTokenizer",
        "model_url": RESTRICTED_ACCESS_URL,
        "intent_tokenization": True,
        "intent_split_symbol": "+",
    }

    tokenizer = ConveRTTokenizer(component_config, ignore_exceptions=True)

    message = Message(data={TEXT: text})
    message.set(INTENT, text)

    tokenizer.train(TrainingData([message]))

    assert [t.text for t in message.get(TOKENS_NAMES[INTENT])] == expected_tokens


@pytest.mark.parametrize(
    "text, expected_number_of_sub_tokens",
    [("Aarhus is a city", [2, 1, 1, 1]), ("sentence embeddings", [1, 3])],
)
def test_convert_tokenizer_number_of_sub_tokens(text, expected_number_of_sub_tokens):
    component_config = {"name": "ConveRTTokenizer", "model_url": RESTRICTED_ACCESS_URL}
    tokenizer = ConveRTTokenizer(component_config, ignore_exceptions=True)

    message = Message(data={TEXT: text})
    message.set(INTENT, text)

    tokenizer.train(TrainingData([message]))

    assert [
        t.get(NUMBER_OF_SUB_TOKENS) for t in message.get(TOKENS_NAMES[TEXT])
    ] == expected_number_of_sub_tokens


def test_raise_no_url():

    component_config = {"name": "ConveRTTokenizer"}
    with pytest.raises(RasaException) as excinfo:
        _ = ConveRTTokenizer(component_config)

    assert (
        """Parameter "model_url" was not specified in the configuration of "ConveRTTokenizer"""
        in str(excinfo.value)
    )


@pytest.mark.parametrize(
    "model_url, exception_phrase",
    [
        (ORIGINAL_TF_HUB_MODULE_URL, "which does not contain the model any longer"),
        (
            RESTRICTED_ACCESS_URL,
            "which is strictly reserved for pytests of Rasa Open Source only",
        ),
    ],
)
def test_raise_invalid_urls(model_url: Text, exception_phrase: Text):

    component_config = {"name": "ConveRTTokenizer", "model_url": model_url}
    with pytest.raises(RasaException) as excinfo:
        _ = ConveRTTokenizer(component_config)

    assert exception_phrase in str(excinfo.value)


def test_raise_wrong_model_directory():

    with tempfile.TemporaryDirectory() as temp_dir:

        component_config = {"name": "ConveRTTokenizer", "model_url": temp_dir}

        with pytest.raises(RasaException) as excinfo:
            _ = ConveRTTokenizer(component_config)

        assert "Re-check the files inside the directory" in str(excinfo.value)


def test_raise_wrong_model_file():

    with tempfile.TemporaryDirectory() as temp_dir:

        # create a dummy file
        temp_file = os.path.join(temp_dir, "saved_model.pb")
        f = open(temp_file, "wb")
        f.close()
        component_config = {"name": "ConveRTTokenizer", "model_url": temp_file}

        with pytest.raises(RasaException) as excinfo:
            _ = ConveRTTokenizer(component_config)

        assert "set to the path of a file which is invalid" in str(excinfo.value)


def test_raise_invalid_path():

    component_config = {"name": "ConveRTTokenizer", "model_url": "saved_model.pb"}

    with pytest.raises(RasaException) as excinfo:
        _ = ConveRTTokenizer(component_config)

    assert "neither a valid remote URL nor a local directory" in str(excinfo.value)
