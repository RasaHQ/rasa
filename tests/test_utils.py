import json

import pytest
from builtins import str
from httpretty import httpretty

from rasa_core import utils
from rasa_core.utils import EndpointConfig


def test_is_int():
    assert utils.is_int(1)
    assert utils.is_int(1.0)
    assert not utils.is_int(None)
    assert not utils.is_int(1.2)
    assert not utils.is_int("test")


def test_subsample_array_read_only():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    r = utils.subsample_array(t, 5,
                              can_modify_incoming_array=False)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_subsample_array():
    t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # this will modify the original array and shuffle it
    r = utils.subsample_array(t, 5)

    assert len(r) == 5
    assert set(r).issubset(t)


def test_on_hot():
    r = utils.one_hot(4, 6)
    assert (r[[0, 1, 2, 3, 5]] == 0).all()
    assert r[4] == 1


def test_on_hot_out_of_range():
    with pytest.raises(ValueError):
        utils.one_hot(4, 3)


def test_list_routes(default_agent):
    from rasa_core import server
    app = server.create_app(default_agent, auth_token=None)

    routes = utils.list_routes(app)
    assert len(routes) > 0


def test_cap_length():
    assert utils.cap_length("mystring", 6) == "mys..."


def test_cap_length_without_ellipsis():
    assert utils.cap_length("mystring", 3,
                            append_ellipsis=False) == "mys"


def test_cap_length_with_short_string():
    assert utils.cap_length("my", 3) == "my"


def test_pad_list_to_size():
    assert utils.pad_list_to_length(["e1", "e2"], 4, "other") == \
           ["e1", "e2", "other", "other"]


def test_read_lines():
    lines = utils.read_lines("data/test_stories/stories.md",
                             max_line_limit=2,
                             line_pattern="\*.*")

    lines = list(lines)

    assert len(lines) == 2


def test_endpoint_config():
    endpoint = EndpointConfig(
            "https://abc.defg/",
            params={"A": "B"},
            headers={"X-Powered-By": "Rasa"},
            basic_auth={"username": "user",
                        "password": "pass"},
            token="mytoken",
            token_name="letoken"
    )

    httpretty.register_uri(
            httpretty.POST,
            'https://abc.defg/test',
            status=500,
            body='')

    httpretty.enable()
    endpoint.request("post", subpath="test",
                     content_type="application/text",
                     json={"c": "d"},
                     params={"P": "1"})
    httpretty.disable()

    r = httpretty.latest_requests[-1]

    assert json.loads(str(r.body.decode("utf-8"))) == {"c": "d"}
    assert r.headers.get("X-Powered-By") == "Rasa"
    assert r.headers.get("Authorization") == "Basic dXNlcjpwYXNz"
    assert r.querystring.get("A") == ["B"]
    assert r.querystring.get("P") == ["1"]
    assert r.querystring.get("letoken") == ["mytoken"]
