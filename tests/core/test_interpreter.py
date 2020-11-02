import pytest
from aioresponses import aioresponses

from rasa.core.interpreter import RasaNLUHttpInterpreter
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import latest_request, json_of_latest_request


@pytest.mark.parametrize(
    "endpoint_url,joined_url",
    [
        ("https://example.com", "https://example.com/model/parse"),
        ("https://example.com/a", "https://example.com/a/model/parse"),
        ("https://example.com/a/", "https://example.com/a/model/parse"),
    ],
)
async def test_http_interpreter(endpoint_url, joined_url):
    with aioresponses() as mocked:
        mocked.post(joined_url)

        endpoint = EndpointConfig(endpoint_url)
        interpreter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        await interpreter.parse(text="message_text", message_id="message_id")

        r = latest_request(mocked, "POST", joined_url)

        query = json_of_latest_request(r)
        response = {"text": "message_text", "token": None, "message_id": "message_id"}

        assert query == response
