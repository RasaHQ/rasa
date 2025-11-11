import textwrap
from pathlib import Path
from typing import Optional, Text, Union
from unittest.mock import Mock

import pytest
import structlog
from aioresponses import aioresponses

import rasa.utils.endpoints as endpoint_utils
from rasa.shared.exceptions import FileNotFoundException
from rasa.tracing.constants import ENDPOINTS_TRACING_KEY
from rasa.tracing.exceptions import DuplicateTracingConfigException
from tests.utilities import json_of_latest_request, latest_request


@pytest.mark.parametrize(
    "base, subpath, expected_result",
    [
        ("https://example.com", None, "https://example.com"),
        ("https://example.com/test", None, "https://example.com/test"),
        ("https://example.com/", None, "https://example.com/"),
        ("https://example.com/", "test", "https://example.com/test"),
        ("https://example.com/", "test/", "https://example.com/test/"),
        (
            "http://duckling.rasa.com:8000",
            "/parse",
            "http://duckling.rasa.com:8000/parse",
        ),
        (
            "http://duckling.rasa.com:8000/",
            "/parse",
            "http://duckling.rasa.com:8000/parse",
        ),
    ],
)
def test_concat_url(base, subpath, expected_result):
    assert endpoint_utils.concat_url(base, subpath) == expected_result


def test_warning_for_base_paths_with_trailing_slash():
    test_path = "base/"
    with structlog.testing.capture_logs() as caplog:
        assert endpoint_utils.concat_url(test_path, None) == test_path

    assert len(caplog) == 1
    assert caplog[0]["event"] == "endpoint.concat_url.trailing_slash"
    assert caplog[0]["log_level"] == "debug"


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


async def test_endpoint_config_with_cafile(tmp_path: Path):
    cafile = "data/test_endpoints/cert.pem"

    with aioresponses() as mocked:
        endpoint = endpoint_utils.EndpointConfig(
            "https://example.com/", cafile=str(cafile)
        )

        mocked.post("https://example.com/", status=200)

        await endpoint.request("post")

        request = latest_request(mocked, "post", "https://example.com/")[-1]

        ssl_context = request.kwargs["ssl"]
        certs = ssl_context.get_ca_certs()
        assert certs[0]["subject"][4][0] == ("organizationalUnitName", "rasa")


async def test_endpoint_config_with_non_existent_cafile(tmp_path: Path):
    cafile = "data/test_endpoints/no_file.pem"

    endpoint = endpoint_utils.EndpointConfig("https://example.com/", cafile=str(cafile))

    with pytest.raises(FileNotFoundException):
        await endpoint.request("post")


def test_endpoint_config_default_token_name():
    test_data = {"url": "http://test", "token": "token"}

    actual = endpoint_utils.EndpointConfig.from_dict(test_data)

    assert actual.token_name == "token"


def test_endpoint_config_custom_token_name():
    test_data = {"url": "http://test", "token": "token", "token_name": "test_token"}

    actual = endpoint_utils.EndpointConfig.from_dict(test_data)

    assert actual.token_name == "test_token"


async def test_request_non_json_response():
    with aioresponses() as mocked:
        endpoint = endpoint_utils.EndpointConfig("https://example.com/")

        mocked.post(
            "https://example.com/test",
            payload="ok",
            content_type="application/text",
            status=200,
        )

        response = await endpoint.request("post", subpath="test")

        assert not response


@pytest.mark.parametrize(
    "filename, endpoint_type",
    [("data/test_endpoints/example_endpoints.yml", "tracker_store")],
)
def test_read_endpoint_config(filename: Text, endpoint_type: Text):
    conf = endpoint_utils.read_endpoint_config(filename, endpoint_type)
    assert isinstance(conf, endpoint_utils.EndpointConfig)


@pytest.mark.parametrize(
    "endpoint_type, cafile",
    [("action_endpoint", "./some_test_file"), ("tracker_store", None)],
)
def test_read_endpoint_config_with_cafile(endpoint_type: Text, cafile: Optional[Text]):
    conf = endpoint_utils.read_endpoint_config(
        "data/test_endpoints/example_endpoints.yml", endpoint_type
    )
    assert conf.cafile == cafile


@pytest.mark.parametrize(
    "filename, endpoint_type",
    [
        ("", "tracker_store"),
        ("data/test_endpoints/example_endpoints.yml", "stuff"),
        ("data/test_endpoints/example_endpoints.yml", "empty"),
        ("/unknown/path.yml", "tracker_store"),
    ],
)
def test_read_endpoint_config_not_found(filename: Text, endpoint_type: Text):
    conf = endpoint_utils.read_endpoint_config(filename, endpoint_type)
    assert conf is None


@pytest.mark.parametrize(
    "value, default, expected_result",
    [
        (None, True, True),
        (False, True, False),
        ("false", True, False),
        ("true", False, True),
    ],
)
def test_bool_arg(
    value: Optional[Union[bool, str]], default: bool, expected_result: bool
):
    request = Mock()
    request.args = {}
    if value is not None:
        request.args = {"key": value}
    assert endpoint_utils.bool_arg(request, "key", default) == expected_result


@pytest.mark.parametrize(
    "value, default, expected_result",
    [(None, 0.5, 0.5), (0.5, None, 0.5), ("0.5", 0, 0.5), ("a", 0.5, 0.5)],
)
def test_float_arg(
    value: Optional[Union[float, str]], default: float, expected_result: float
):
    request = Mock()
    request.args = {}
    if value is not None:
        request.args = {"key": value}
    assert endpoint_utils.float_arg(request, "key", default) == expected_result


@pytest.mark.parametrize(
    "value, default, expected_result",
    [(None, 0, 0), (1, 0, 1), ("1", 0, 1), ("a", 0, 0)],
)
def test_int_arg(value: Optional[Union[int, str]], default: int, expected_result: int):
    request = Mock()
    request.args = {}
    if value is not None:
        request.args = {"key": value}
    assert endpoint_utils.int_arg(request, "key", default) == expected_result


# ------------------------------------------------------------
# read_backend_tracing_configuration tests
# ------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("", None),
        (None, None),
    ],
)
def test_read_backend_tracing_configuration_empty_filename(
    filename: str, expected: Optional[Text]
) -> None:
    """Test that read_backend_tracing_configuration returns None for
    empty/None filename.
    """
    result = endpoint_utils.read_backend_tracing_configuration(
        filename, ENDPOINTS_TRACING_KEY
    )
    assert result == expected


def test_read_backend_tracing_configuration_file_not_found() -> None:
    """Test that read_backend_tracing_configuration returns None when file is not
    found."""
    result = endpoint_utils.read_backend_tracing_configuration(
        "/nonexistent/path/endpoints.yml", ENDPOINTS_TRACING_KEY
    )
    assert result is None


def test_read_backend_tracing_configuration_no_tracing_key(
    tmp_path: Path,
) -> None:
    """Test that read_backend_tracing_configuration returns None when tracing
    key is missing.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            action_endpoint:
                url: "http://localhost:5056/webhook"
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is None


def test_read_backend_tracing_configuration_config_is_none(
    tmp_path: Path,
) -> None:
    """Test that read_backend_tracing_configuration returns None when config value
    is None."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing: null
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is None


def test_read_backend_tracing_configuration_single_dict(tmp_path: Path) -> None:
    """Test that read_backend_tracing_configuration returns config when single
    dict provided.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: jaeger
                host: localhost
                port: 6831
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is not None
    assert result.type == "jaeger"
    assert result.kwargs["host"] == "localhost"
    assert result.kwargs["port"] == 6831


def test_read_backend_tracing_configuration_list_single_config(tmp_path: Path) -> None:
    """Test that read_backend_tracing_configuration returns config when list with
    one config.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                - type: jaeger
                  host: localhost
                  port: 6831
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is not None
    assert result.type == "jaeger"
    assert result.kwargs["host"] == "localhost"


def test_read_backend_tracing_configuration_multiple_configs(tmp_path: Path) -> None:
    """Test that read_backend_tracing_configuration raises exception for
    multiple configs.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                - type: jaeger
                  host: localhost
                  port: 6831
                - type: otlp
                  endpoint: http://localhost:4317
            """
        )
    )
    with pytest.raises(DuplicateTracingConfigException) as exc_info:
        endpoint_utils.read_backend_tracing_configuration(
            str(endpoints_file), ENDPOINTS_TRACING_KEY
        )
    assert "Multiple tracing configs found" in str(exc_info.value)


def test_read_backend_tracing_configuration_filters_langfuse(
    tmp_path: Path,
) -> None:
    """Test that read_backend_tracing_configuration filters out langfuse configs."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                - type: langfuse
                  public_key: $LANGFUSE_PUBLIC_KEY
                  private_key: $LANGFUSE_SECRET_KEY
                  host: https://cloud.langfuse.com
                - type: jaeger
                  host: localhost
                  port: 6831
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is not None
    assert result.type == "jaeger"
    assert result.kwargs["host"] == "localhost"


def test_read_backend_tracing_configuration_only_langfuse(
    tmp_path: Path,
) -> None:
    """Test that read_backend_tracing_configuration returns None when only
    langfuse config exists.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: $LANGFUSE_PUBLIC_KEY
                private_key: $LANGFUSE_SECRET_KEY
                host: https://cloud.langfuse.com
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is None


def test_read_backend_tracing_configuration_langfuse_in_list(
    tmp_path: Path,
) -> None:
    """Test that read_backend_tracing_configuration filters langfuse from list."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                - type: langfuse
                  public_key: $LANGFUSE_PUBLIC_KEY
                  private_key: $LANGFUSE_SECRET_KEY
                  host: https://cloud.langfuse.com
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is None


def test_read_backend_tracing_configuration_otlp_config(tmp_path: Path) -> None:
    """Test that read_backend_tracing_configuration handles otlp config correctly."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: otlp
                endpoint: http://localhost:4317
                service_name: test-service
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is not None
    assert result.type == "otlp"
    assert result.kwargs["endpoint"] == "http://localhost:4317"
    assert result.kwargs["service_name"] == "test-service"


def test_read_backend_tracing_configuration_multiple_after_filtering(
    tmp_path: Path,
) -> None:
    """Test that read_backend_tracing_configuration raises exception when
    multiple non-langfuse configs exist.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                - type: langfuse
                  public_key: $LANGFUSE_PUBLIC_KEY
                  private_key: $LANGFUSE_SECRET_KEY
                  host: https://cloud.langfuse.com
                - type: jaeger
                  host: localhost
                  port: 6831
                - type: otlp
                  endpoint: http://localhost:4317
            """
        )
    )
    with pytest.raises(DuplicateTracingConfigException) as exc_info:
        endpoint_utils.read_backend_tracing_configuration(
            str(endpoints_file), ENDPOINTS_TRACING_KEY
        )
    assert "Multiple tracing configs found" in str(exc_info.value)


def test_read_backend_tracing_configuration_empty_list(tmp_path: Path) -> None:
    """Test that read_backend_tracing_configuration returns None when config is
    empty list."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing: []
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), ENDPOINTS_TRACING_KEY
    )
    assert result is None


def test_read_backend_tracing_configuration_custom_endpoint_type(
    tmp_path: Path,
) -> None:
    """Test that read_backend_tracing_configuration works with custom endpoint types."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            custom_endpoint:
                type: jaeger
                host: localhost
                port: 6831
            """
        )
    )
    result = endpoint_utils.read_backend_tracing_configuration(
        str(endpoints_file), "custom_endpoint"
    )
    assert result is not None
    assert result.type == "jaeger"
    assert result.kwargs["host"] == "localhost"
