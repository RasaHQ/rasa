from unittest.mock import patch

import pytest
from aioresponses import aioresponses
from rasa.core.channels import UserMessage
from rasa.core.http_interpreter import RasaNLUHttpInterpreter
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import json_of_latest_request, latest_request


@pytest.mark.parametrize(
    "endpoint_url,joined_url",
    [
        ("https://example.com", "https://example.com/model/parse"),
        ("https://example.com/a", "https://example.com/a/model/parse"),
        ("https://example.com/a/", "https://example.com/a/model/parse"),
    ],
)
async def test_http_interpreter(endpoint_url, joined_url):
    """
    GIVEN an endpoint url
    WHEN a RasaNLUHttpInterpreter is created using the endpoint url
    THEN the parse method sends a request to the joined url.
    """
    with aioresponses() as mocked:
        mocked.post(joined_url)

        endpoint = EndpointConfig(endpoint_url)
        interpreter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        message = UserMessage(text="message_text", sender_id="message_id")
        await interpreter.parse(message)

        r = latest_request(mocked, "POST", joined_url)

        query = json_of_latest_request(r)
        response = {"text": "message_text", "token": None, "message_id": "message_id"}

        assert query == response


@pytest.fixture
def nlu_interpreter():
    with patch("aiohttp.ClientSession") as mock_session:
        endpoint = EndpointConfig("https://example.com/a/")
        yield RasaNLUHttpInterpreter(endpoint_config=endpoint)

        # Assert that the session object is initialized correctly
        assert mock_session.called


async def test_same_session_object_used(nlu_interpreter):
    """
    GIVEN a RasaNLUHttpInterpreter
    WHEN the parse() method is called multiple times
    THEN the same session object is used for all requests.
    """
    # Call the parse() method multiple times
    session = nlu_interpreter.session

    result1 = await nlu_interpreter.parse(
        UserMessage(text="message_text_1", sender_id="message_id_1")
    )
    assert nlu_interpreter.session == session

    result2 = await nlu_interpreter.parse(
        UserMessage(text="message_text_2", sender_id="message_id_2")
    )
    assert nlu_interpreter.session == session

    # Assert that the same session object is used for all requests
    assert result1 is not None
    assert result2 is not None
