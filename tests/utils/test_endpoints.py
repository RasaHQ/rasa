import logging

import pytest
from aioresponses import aioresponses

from tests.utilities import latest_request, json_of_latest_request
import rasa.utils.endpoints as endpoint_utils


@pytest.mark.parametrize(
    "base, subpath, expected_result",
    [
        ("https://example.com", None, "https://example.com"),
        ("https://example.com/test", None, "https://example.com/test"),
        ("https://example.com/", None, "https://example.com/"),
        ("https://example.com/", "test", "https://example.com/test"),
        ("https://example.com/", "test/", "https://example.com/test/"),
    ],
)
def test_concat_url(base, subpath, expected_result):
    assert endpoint_utils.concat_url(base, subpath) == expected_result


def test_warning_for_base_paths_with_trailing_slash(caplog):
    test_path = "base/"

    with caplog.at_level(logging.DEBUG, logger="rasa.utils.endpoints"):
        assert endpoint_utils.concat_url(test_path, None) == test_path

    assert len(caplog.records) == 1


async def test_endpoint_config():
    with aioresponses() as mocked:
        endpoint = endpoint_utils.EndpointConfig(
            "https://example.com/",
            params={"A": "B"},
            headers={"X-Powered-By": "Rasa"},
            basic_auth={"username": "user", "password": "pass"},
            token="mytoken",
            token_name="letoken",
            type="redis",
            port=6379,
            db=0,
            password="password",
            timeout=30000,
        )

        mocked.post(
            "https://example.com/test?A=B&P=1&letoken=mytoken",
            payload={"ok": True},
            repeat=True,
            status=200,
        )

        await endpoint.request(
            "post",
            subpath="test",
            content_type="application/text",
            json={"c": "d"},
            params={"P": "1"},
        )

        r = latest_request(
            mocked, "post", "https://example.com/test?A=B&P=1&letoken=mytoken"
        )

        assert r

        assert json_of_latest_request(r) == {"c": "d"}
        assert r[-1].kwargs.get("params", {}).get("A") == "B"
        assert r[-1].kwargs.get("params", {}).get("P") == "1"
        assert r[-1].kwargs.get("params", {}).get("letoken") == "mytoken"

        # unfortunately, the mock library won't report any headers stored on
        # the session object, so we need to verify them separately
        async with endpoint.session() as s:
            assert s._default_headers.get("X-Powered-By") == "Rasa"
            assert s._default_auth.login == "user"
            assert s._default_auth.password == "pass"


def test_endpoint_config_default_token_name():
    test_data = {"url": "http://test", "token": "token"}

    actual = endpoint_utils.EndpointConfig.from_dict(test_data)

    assert actual.token_name == "token"


def test_endpoint_config_custom_token_name():
    test_data = {"url": "http://test", "token": "token", "token_name": "test_token"}

    actual = endpoint_utils.EndpointConfig.from_dict(test_data)

    assert actual.token_name == "test_token"
