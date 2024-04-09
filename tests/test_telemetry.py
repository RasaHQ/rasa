import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Text

import yaml
from pytest import MonkeyPatch, LogCaptureFixture
from unittest.mock import MagicMock, Mock, patch
import pytest
import responses

from rasa import telemetry
import rasa.constants
import rasa.utils.licensing
from rasa.dialogue_understanding.generator.llm_command_generator import (
    DEFAULT_LLM_CONFIG as LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG,
)
from rasa.dialogue_understanding.generator.flow_retrieval import (
    DEFAULT_EMBEDDINGS_CONFIG,
)

from rasa.e2e_test.e2e_test_case import TestCase, Fixture
from rasa.telemetry import (
    SEGMENT_IDENTIFY_ENDPOINT,
    SEGMENT_REQUEST_TIMEOUT,
    SEGMENT_TRACK_ENDPOINT,
    TELEMETRY_E2E_TEST_RUN_STARTED_EVENT,
    TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE,
    TELEMETRY_ID,
    TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE,
    TRACING_BACKEND,
    LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED,
    FLOW_RETRIEVAL_ENABLED,
    FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME,
    LLM_COMMAND_GENERATOR_MODEL_NAME,
    _get_llm_command_generator_config,
)
from rasa.utils.licensing import LICENSE_ENV_VAR
from tests.tracing.conftest import TRACING_TESTS_FIXTURES_DIRECTORY

TELEMETRY_TEST_USER = "083642a3e448423ca652134f00e7fc76"  # just some random static id
TELEMETRY_TEST_KEY = "5640e893c1324090bff26f655456caf3"  # just some random static id


@pytest.fixture(autouse=True)
def patch_global_config_path(tmp_path: Path) -> Generator[None, None, None]:
    """Ensure we use a unique config path for each test to avoid tests influencing
    each other."""
    default_location = rasa.constants.GLOBAL_USER_CONFIG_PATH
    rasa.constants.GLOBAL_USER_CONFIG_PATH = str(tmp_path / "global.yml")
    yield
    rasa.constants.GLOBAL_USER_CONFIG_PATH = default_location


@pytest.fixture(autouse=True)
def patch_telemetry_context() -> Generator[None, None, None]:
    """Use a new telemetry context for each test to avoid tests influencing each other."""
    defaut_context = telemetry.TELEMETRY_CONTEXT
    telemetry.TELEMETRY_CONTEXT = None
    yield
    telemetry.TELEMETRY_CONTEXT = defaut_context


async def _mock_track_internal_exception(*args, **kwargs) -> None:
    raise Exception("Tracking failed")


def test_config_path_empty(monkeypatch: MonkeyPatch):
    # this tests the patch_global_config_path fixture -> makes sure the config
    # is read from a temp file instead of the default location
    assert "/.config/rasa" not in rasa.constants.GLOBAL_USER_CONFIG_PATH


def test_segment_request_header():
    assert telemetry.segment_request_header(TELEMETRY_TEST_KEY) == {
        "Content-Type": "application/json",
        "Authorization": "Basic NTY0MGU4OTNjMTMyNDA5MGJmZjI2ZjY1NTQ1NmNhZjM6",
    }


def test_segment_payload():
    assert telemetry.segment_request_payload(
        TELEMETRY_TEST_USER, "foobar", {"foo": "bar"}, {}
    ) == {
        "userId": TELEMETRY_TEST_USER,
        "event": "foobar",
        "properties": {"foo": "bar"},
        "context": {},
    }


def test_track_ignore_exception(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(telemetry, "_send_event", _mock_track_internal_exception)

    # If the test finishes without raising any exceptions, then it's successful
    assert telemetry._track("Test") is None


def test_initialize_telemetry():
    telemetry.initialize_telemetry()
    assert True


def test_initialize_telemetry_with_env_false(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "false")
    assert telemetry.initialize_telemetry() is False


def test_initialize_telemetry_with_env_true(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")
    assert telemetry.initialize_telemetry() is True


def test_initialize_telemetry_env_overwrites_config(monkeypatch: MonkeyPatch):
    telemetry.toggle_telemetry_reporting(True)
    assert telemetry.initialize_telemetry() is True

    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "false")
    assert telemetry.initialize_telemetry() is False


def test_initialize_telemetry_prints_info(monkeypatch: MonkeyPatch):
    # Mock actual training
    mock = Mock()
    monkeypatch.setattr(telemetry, "print_telemetry_reporting_info", mock)

    telemetry.initialize_telemetry()

    mock.assert_called_once()


def test_not_in_ci_if_not_in_ci(monkeypatch: MonkeyPatch):
    for env in telemetry.CI_ENVIRONMENT_TELL:
        monkeypatch.delenv(env, raising=False)

    assert not telemetry.in_continuous_integration()


def test_in_ci_if_in_ci(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CI", "true")

    assert telemetry.in_continuous_integration()


def test_with_default_context_fields_contains_package_versions():
    context = telemetry.with_default_context_fields()
    assert "python" in context
    assert context["rasa_open_source"] == rasa.__version__


def test_default_context_fields_overwrite_by_context():
    context = telemetry.with_default_context_fields({"python": "foobar"})
    assert context["python"] == "foobar"


def test_track_sends_telemetry_id(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")
    telemetry.initialize_telemetry()

    mock = Mock()
    monkeypatch.setattr(telemetry, "_send_event", mock)
    telemetry._track("foobar", {"foo": "bar"}, {"baz": "foo"})

    assert telemetry.get_telemetry_id() is not None

    mock.assert_called_once()
    call_args = mock.call_args[0]

    assert call_args[0] == telemetry.get_telemetry_id()
    assert call_args[1] == "foobar"
    assert call_args[2]["foo"] == "bar"
    assert call_args[2]["metrics_id"] == telemetry.get_telemetry_id()
    assert call_args[3]["baz"] == "foo"


def test_toggle_telemetry_reporting(monkeypatch: MonkeyPatch):
    # tests that toggling works if there is no config
    telemetry.toggle_telemetry_reporting(True)
    assert telemetry.initialize_telemetry() is True

    telemetry.toggle_telemetry_reporting(False)
    assert telemetry.initialize_telemetry() is False

    # tests that toggling works if config is set to false
    telemetry.toggle_telemetry_reporting(True)
    assert telemetry.initialize_telemetry() is True


def test_segment_gets_called(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_WRITE_KEY", "foobar")
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")
    telemetry.initialize_telemetry()

    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, "https://api.segment.io/v1/track", json={})

        telemetry._track(
            "test event", {"foo": "bar"}, {"foobar": "baz", "license_hash": "foobar"}
        )

        assert len(rsps.calls) == 1
        r = rsps.calls[0]

        assert r
        b = json.loads(r.request.body)

        assert "userId" in b
        assert b["event"] == "test event"
        assert b["properties"].get("foo") == "bar"
        assert b["context"].get("foobar") == "baz"


def test_segment_does_not_raise_exception_on_failure(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("RASA_TELEMETRY_WRITE_KEY", "foobar")
    telemetry.initialize_telemetry()

    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, "https://api.segment.io/v1/track", body="", status=505)

        # this call should complete without throwing an exception
        telemetry._track(
            "test event", {"foo": "bar"}, {"foobar": "baz", "license_hash": "foobar"}
        )

        assert rsps.assert_call_count("https://api.segment.io/v1/track", 1)


def test_segment_does_not_get_called_without_license(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")
    monkeypatch.setenv("RASA_TELEMETRY_WRITE_KEY", "foobar")

    def mock_get_license_hash(*args, **kwargs):
        return None

    monkeypatch.setattr(
        rasa.telemetry.plugin_manager().hook, "get_license_hash", mock_get_license_hash
    )

    mock_license_property = MagicMock(return_value=None)
    monkeypatch.setattr(
        rasa.telemetry, "property_of_active_license", mock_license_property
    )

    telemetry.initialize_telemetry()

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        rsps.add(responses.POST, "https://api.segment.io/v1/track", body="", status=505)

        # this call should complete without throwing an exception
        telemetry._track("test event", {"foo": "bar"}, {"foobar": "baz"})

        assert rsps.assert_call_count("https://api.segment.io/v1/track", 0)


def test_environment_write_key_overwrites_key_file(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_WRITE_KEY", "foobar")
    assert telemetry.telemetry_write_key() == "foobar"


def test_sentry_event_pii_removal():
    # this is an example event taken from sentry (generated by putting a print
    # into `telemetry.strip_sensitive_data_from_sentry_event`)
    event = {
        "level": "error",
        "exception": {
            "values": [
                {
                    "module": None,
                    "type": "Exception",
                    "value": "Some unexpected exception.",
                    "mechanism": {"type": "excepthook", "handled": False},
                    "stacktrace": {
                        "frames": [
                            {
                                "filename": "rasa",
                                "abs_path": "/Users/tmbo/Library/Caches/pypoetry/virtualenvs/rasa-U5VQkfdm-py3.6/bin/rasa",
                                "function": "<module>",
                                "module": "__main__",
                                "lineno": 33,
                                "pre_context": [
                                    "globals().setdefault('load_entry_point', importlib_load_entry_point)",
                                    "",
                                    "",
                                    "if __name__ == '__main__':",
                                    "    sys.argv[0] = re.sub(r'(-script\\.pyw?|\\.exe)?$', '', sys.argv[0])",
                                ],
                                "context_line": "    sys.exit(load_entry_point('rasa', 'console_scripts', 'rasa')())",
                                "post_context": [],
                            },
                            {
                                "filename": "rasa/__main__.py",
                                "abs_path": "/Users/tmbo/lastmile/bot-ai/rasa/rasa/__main__.py",
                                "function": "main",
                                "module": "rasa.__main__",
                                "lineno": 113,
                                "pre_context": [
                                    "",
                                    '    if hasattr(cmdline_arguments, "func"):',
                                    "        rasa.utils.io.configure_colored_logging(log_level)",
                                    "        set_log_and_warnings_filters()",
                                    "        rasa.telemetry.initialize_error_reporting()",
                                ],
                                "context_line": "        cmdline_arguments.func(cmdline_arguments)",
                                "post_context": [
                                    '    elif hasattr(cmdline_arguments, "version"):',
                                    "        print_version()",
                                    "    else:",
                                    "        # user has not provided a subcommand, let's print the help",
                                    '        logger.error("No command specified.")',
                                ],
                                "in_app": True,
                            },
                            {
                                "filename": "rasa/cli/train.py",
                                "abs_path": "/Users/tmbo/lastmile/bot-ai/rasa/rasa/cli/train.py",
                                "function": "train",
                                "module": "rasa.cli.train",
                                "lineno": 69,
                                "pre_context": [
                                    "    training_files = [",
                                    '        get_validated_path(f, "data", DEFAULT_DATA_PATH, none_is_valid=True)',
                                    "        for f in args.data",
                                    "    ]",
                                    "",
                                ],
                                "context_line": '    raise Exception("Some unexpected exception.")',
                                "post_context": [
                                    "",
                                    "    return rasa.train(",
                                    "        domain=domain,",
                                    "        config=config,",
                                    "        training_files=training_files,",
                                ],
                                "in_app": True,
                            },
                        ]
                    },
                }
            ]
        },
        "event_id": "73dd4980a5fd498d96fec2ee3ee0cb86",
        "timestamp": "2020-09-14T14:37:14.237740Z",
        "breadcrumbs": {"values": []},
        "release": "rasa-2.0.0a4",
        "environment": "production",
        "server_name": "99ec342261934892aac1784d1ac061c1",
        "sdk": {
            "name": "sentry.python",
            "version": "0.17.5",
            "packages": [{"name": "pypi:sentry-sdk", "version": "0.17.5"}],
            "integrations": ["atexit", "dedupe", "excepthook"],
        },
        "platform": "python",
    }
    stripped = telemetry.strip_sensitive_data_from_sentry_event(event)

    for value in stripped.get("exception", {}).get("values", []):
        for frame in value.get("stacktrace", {}).get("frames", []):
            # make sure absolute path got removed from all stack entries
            assert not frame.get("abs_path")


def _create_exception_event_in_file(filename: Text) -> Dict[Text, Any]:
    """Create a sentry error event with the filename as the file the error occurred in.

    Args:
        filename: name of the file the mock error supposedly happened in
    Returns:
        mock sentry error event
    """
    # this is an example event taken from sentry (generated by putting a print
    # into `telemetry.strip_sensitive_data_from_sentry_event`)
    return {
        "level": "error",
        "exception": {
            "values": [
                {
                    "module": None,
                    "type": "Exception",
                    "value": "Some unexpected exception.",
                    "mechanism": {"type": "excepthook", "handled": False},
                    "stacktrace": {
                        "frames": [
                            {
                                "filename": filename,
                                "abs_path": "/Users/tmbo/Library/Caches/pypoetry/virtualenvs/rasa-U5VQkfdm-py3.6/bin/rasa",
                                "function": "<module>",
                                "module": "__main__",
                                "lineno": 33,
                                "pre_context": [
                                    "globals().setdefault('load_entry_point', importlib_load_entry_point)",
                                    "",
                                    "",
                                    "if __name__ == '__main__':",
                                    "    sys.argv[0] = re.sub(r'(-script\\.pyw?|\\.exe)?$', '', sys.argv[0])",
                                ],
                                "context_line": "    sys.exit(load_entry_point('rasa', 'console_scripts', 'rasa')())",
                                "post_context": [],
                            }
                        ]
                    },
                }
            ]
        },
        "event_id": "73dd4980a5fd498d96fec2ee3ee0cb86",
        "timestamp": "2020-09-14T14:37:14.237740Z",
        "breadcrumbs": {"values": []},
        "release": "rasa-2.0.0a4",
        "environment": "production",
        "server_name": "99ec342261934892aac1784d1ac061c1",
        "sdk": {
            "name": "sentry.python",
            "version": "0.17.5",
            "packages": [{"name": "pypi:sentry-sdk", "version": "0.17.5"}],
            "integrations": ["atexit", "dedupe", "excepthook"],
        },
        "platform": "python",
    }


def test_sentry_drops_error_in_custom_path():
    event = _create_exception_event_in_file("/my_project/mymodule.py")
    stripped = telemetry.strip_sensitive_data_from_sentry_event(event)

    assert stripped is None


def test_sentry_works_fine_with_relative_paths():
    event = _create_exception_event_in_file("rasa/train.py")
    stripped = telemetry.strip_sensitive_data_from_sentry_event(event)

    assert stripped is not None

    stack_frames = stripped["exception"]["values"][0]["stacktrace"]["frames"]
    assert stack_frames[0]["filename"] == "rasa/train.py"


def test_sentry_strips_absolute_path_from_site_packages():
    event = _create_exception_event_in_file(
        "/Users/tmbo/Library/Caches/pypoetry/virtualenvs/rasa-U5VQkfdm-py3.7/lib/python3.7/site-packages/rasa/train.py"
    )
    stripped = telemetry.strip_sensitive_data_from_sentry_event(event)

    assert stripped is not None

    stack_frames = stripped["exception"]["values"][0]["stacktrace"]["frames"]
    assert stack_frames[0]["filename"] == f"site-packages{os.path.sep}rasa/train.py"


def test_sentry_strips_absolute_path_from_dist_packages():
    event = _create_exception_event_in_file(
        "C:\\Users\\tmbo\\AppData\\Roaming\\Python\\Python35\\dist-packages\\rasa\\train.py"
    )
    stripped = telemetry.strip_sensitive_data_from_sentry_event(event)

    assert stripped is not None

    stack_frames = stripped["exception"]["values"][0]["stacktrace"]["frames"]
    assert stack_frames[0]["filename"] == f"dist-packages{os.path.sep}rasa\\train.py"


def test_context_contains_os():
    context = telemetry._default_context_fields()

    assert "os" in context

    context.pop("os")

    assert "os" in telemetry._default_context_fields()


def test_context_contains_license_hash(monkeypatch: MonkeyPatch) -> None:
    mock = MagicMock()
    mock.return_value.hook.get_license_hash.return_value = "1234567890"
    monkeypatch.setattr("rasa.telemetry.plugin_manager", mock)
    context = telemetry._default_context_fields()

    assert "license_hash" in context
    assert mock.return_value.hook.get_license_hash.called
    assert context["license_hash"] == "1234567890"

    # make sure it is still there after removing it
    context.pop("license_hash")
    assert "license_hash" in telemetry._default_context_fields()


def test_context_does_not_contain_license_hash(monkeypatch: MonkeyPatch) -> None:
    mock = MagicMock()
    mock.return_value.hook.get_license_hash.return_value = None
    monkeypatch.setattr("rasa.telemetry.plugin_manager", mock)
    context = telemetry._default_context_fields()

    assert "license_hash" not in context
    assert mock.return_value.hook.get_license_hash.called


def test_segment_identify_payload() -> None:
    assert telemetry.segment_identify_request_payload(
        TELEMETRY_TEST_USER, {"foo": "bar"}, {}
    ) == {
        "userId": TELEMETRY_TEST_USER,
        "traits": {"foo": "bar"},
        "context": {},
    }


def test_identify_ignore_exception(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry, "_send_traits", _mock_track_internal_exception)

    # If the test finishes without raising any exceptions, then it's successful
    try:
        telemetry._identify({})
    except Exception:
        pytest.fail("Exception was not ignored during a telemetry identify call.")


def test_identify_sends_telemetry_id(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")
    telemetry.initialize_telemetry()

    mock = Mock()
    monkeypatch.setattr(telemetry, "_send_traits", mock)
    telemetry._identify({"foo": "bar"}, {"baz": "foo"})

    assert telemetry.get_telemetry_id() is not None

    mock.assert_called_once()
    call_args = mock.call_args[0]

    assert call_args[0] == telemetry.get_telemetry_id()
    assert call_args[1]["foo"] == "bar"
    assert call_args[2]["baz"] == "foo"


@pytest.mark.parametrize(
    "tracing_backend, endpoints_file",
    [
        (
            "jaeger",
            "jaeger_endpoints.yml",
        )
    ],
)
def test_segment_gets_called_for_identify(
    tracing_backend: Text,
    endpoints_file: Text,
    valid_license: Text,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)
    monkeypatch.setenv(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE, "foobar")
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")
    endpoints_file = str(TRACING_TESTS_FIXTURES_DIRECTORY / endpoints_file)
    telemetry.initialize_telemetry()

    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, SEGMENT_IDENTIFY_ENDPOINT, body="", json={})

        telemetry.identify_endpoint_config_traits(
            endpoints_file, context={"foobar": "baz"}
        )

        assert len(rsps.calls) == 1
        r = rsps.calls[0]

        assert r
        assert isinstance(r, responses.Call)

        assert r.request.body is not None
        b = json.loads(r.request.body)

        assert "userId" in b
        assert b["traits"][TRACING_BACKEND] == tracing_backend
        assert (
            b["context"]["license_hash"]
            == hashlib.sha256(valid_license.encode("utf-8")).hexdigest()
        )
        assert b["context"].get("foobar") == "baz"


def test_segment_identify_does_not_raise_exception_on_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")
    monkeypatch.setenv(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE, "foobar")
    telemetry.initialize_telemetry()

    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, SEGMENT_IDENTIFY_ENDPOINT, body="", status=505)

        # this call should complete without throwing an exception
        telemetry._identify({"foo": "bar"}, {"foobar": "baz"})

        assert rsps.assert_call_count(SEGMENT_IDENTIFY_ENDPOINT, 1)


def test_identify_sets_default_traits(
    monkeypatch: MonkeyPatch, valid_license: Text
) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)
    monkeypatch.setenv(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE, "default")
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")
    telemetry.initialize_telemetry()

    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, SEGMENT_IDENTIFY_ENDPOINT, body="", json={})

        telemetry.identify_endpoint_config_traits(None)

        assert len(rsps.calls) == 1
        r = rsps.calls[0]

        assert r
        assert isinstance(r, responses.Call)

        assert r.request.body is not None
        b = json.loads(r.request.body)

        assert "userId" in b
        assert b["traits"][TRACING_BACKEND] is None
        assert (
            b["context"]["license_hash"]
            == hashlib.sha256(valid_license.encode("utf-8")).hexdigest()
        )


def test_get_telemetry_id_valid(monkeypatch: MonkeyPatch, valid_license: Text) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)

    assert telemetry.get_telemetry_id() is not None


def test_get_telemetry_id_no_license(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(LICENSE_ENV_VAR)

    assert telemetry.get_telemetry_id() is None


def test_get_telemetry_id_invalid(
    monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, "some_invalid_string")

    with caplog.at_level(logging.WARNING):
        assert telemetry.get_telemetry_id() is None

    assert all(
        ["The provided license is invalid" in message for message in caplog.messages]
    )


@pytest.mark.parametrize(
    "input_test_cases, input_fixtures, expected_number_of_test_cases, "
    "expected_number_of_fixtures, expected_uses_fixtures",
    [
        (
            [
                TestCase(name="case 1", steps=[]),
                TestCase(name="case 2", steps=[]),
                TestCase(name="case 3", steps=[]),
            ],
            [
                Fixture(name="fixture 1", slots_set={}),
                Fixture(name="fixture 2", slots_set={}),
            ],
            3,
            2,
            True,
        ),
        (
            [],
            [
                Fixture(name="fixture 1", slots_set={}),
                Fixture(name="fixture 2", slots_set={}),
            ],
            0,
            2,
            True,
        ),
        (
            [
                TestCase(name="case 1", steps=[]),
                TestCase(name="case 2", steps=[]),
                TestCase(name="case 3", steps=[]),
            ],
            [],
            3,
            0,
            False,
        ),
    ],
)
@patch("rasa.telemetry._track")
def test_track_e2e_test_run(
    mock_track: MagicMock,
    input_test_cases: List["TestCase"],
    input_fixtures: List["Fixture"],
    expected_number_of_test_cases: int,
    expected_number_of_fixtures: int,
    expected_uses_fixtures: bool,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    telemetry.track_e2e_test_run(input_test_cases, input_fixtures)

    mock_track.assert_called_once_with(
        TELEMETRY_E2E_TEST_RUN_STARTED_EVENT,
        {
            "number_of_test_cases": expected_number_of_test_cases,
            "number_of_fixtures": expected_number_of_fixtures,
            "uses_fixtures": expected_uses_fixtures,
        },
    )


@pytest.mark.parametrize(
    "event_name, properties, context, telemetry_id, expected_properties",
    [
        (
            "event",
            {"foo": "bar"},
            {"some_ctx_field_1": "some_ctx_value_1"},
            "some_id",
            {"foo": "bar", TELEMETRY_ID: "some_id"},
        ),
        (
            "event",
            {},
            {"some_ctx_field_1": "some_ctx_value_1"},
            "some_id",
            {TELEMETRY_ID: "some_id"},
        ),
    ],
)
@patch("rasa.telemetry.with_default_context_fields")
@patch("rasa.telemetry._send_event")
@patch("rasa.telemetry.get_telemetry_id")
def test_track(
    mock_get_telemetry_id: MagicMock,
    mock_send_event: MagicMock,
    mock_with_default_context_fields: MagicMock,
    event_name: Text,
    properties: Dict[Text, Any],
    context: Dict[Text, Any],
    telemetry_id: Optional[Text],
    expected_properties: Dict[Text, Any],
) -> None:
    mock_get_telemetry_id.return_value = telemetry_id
    mock_with_default_context_fields.return_value = context
    telemetry._track(event_name=event_name, properties=properties, context=context)

    mock_get_telemetry_id.assert_called_once()
    mock_send_event.assert_called_once_with(
        telemetry_id,
        event_name,
        expected_properties,
        context,
    )


@pytest.fixture
def mock_get_telemetry_id(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_get_telemetry_id = MagicMock()
    mock_get_telemetry_id.return_value = None
    monkeypatch.setattr("rasa.telemetry.get_telemetry_id", mock_get_telemetry_id)
    return mock_get_telemetry_id


@pytest.fixture
def mock_send_event(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_send_event = MagicMock()
    monkeypatch.setattr("rasa.telemetry._send_event", mock_send_event)
    return mock_send_event


@pytest.fixture
def mock_with_default_context_fields(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_with_default_context_fields = MagicMock()
    monkeypatch.setattr(
        "rasa.telemetry.with_default_context_fields",
        mock_with_default_context_fields,
    )
    return mock_with_default_context_fields


def test_track_no_event_name(
    mock_send_event: MagicMock,
    mock_with_default_context_fields: MagicMock,
    mock_get_telemetry_id: MagicMock,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_get_telemetry_id.return_value = None

    with caplog.at_level(logging.DEBUG):
        telemetry._track(event_name="event", properties={}, context={})

    mock_get_telemetry_id.assert_called_once()
    mock_send_event.assert_not_called()
    mock_with_default_context_fields.assert_not_called()

    log_msg = "Will not report telemetry events as no ID was found."
    assert log_msg in caplog.text


@pytest.fixture
def mock_segment_track_request_payload(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_segment_track_request_payload = MagicMock()
    monkeypatch.setattr(
        "rasa.telemetry.segment_request_payload",
        mock_segment_track_request_payload,
    )
    return mock_segment_track_request_payload


@pytest.fixture
def mock_send_request(
    monkeypatch: MonkeyPatch,
) -> MagicMock:
    mock_send_request = MagicMock()
    monkeypatch.setattr("rasa.telemetry._send_request", mock_send_request)
    return mock_send_request


def test_send_event(
    mock_segment_track_request_payload: MagicMock,
    mock_send_request: MagicMock,
) -> None:
    payload = {
        "event": "some_event",
        "properties": {"some_prop": "some_value"},
        "context": {"some_ctx_field": "some_ctx_value"},
    }

    mock_segment_track_request_payload.return_value = payload
    telemetry._send_event(
        distinct_id="some_id",
        event_name="some_event",
        properties={"some_prop": "some_value"},
        context={"some_ctx_field": "some_ctx_value"},
    )

    mock_segment_track_request_payload.assert_called_once_with(
        "some_id",
        "some_event",
        {"some_prop": "some_value"},
        {"some_ctx_field": "some_ctx_value"},
    )

    mock_send_request.assert_called_once_with(SEGMENT_TRACK_ENDPOINT, payload)


@pytest.fixture
def mock_is_telemetry_debug_enabled(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_is_telemetry_debug_enabled = MagicMock()
    monkeypatch.setattr(
        "rasa.telemetry._is_telemetry_debug_enabled",
        mock_is_telemetry_debug_enabled,
    )
    return mock_is_telemetry_debug_enabled


@pytest.fixture
def mock_print_telemetry_payload(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_print_telemetry_payload = MagicMock()
    monkeypatch.setattr(
        "rasa.telemetry.print_telemetry_payload",
        mock_print_telemetry_payload,
    )
    return mock_print_telemetry_payload


@pytest.fixture
def mock_get_telemetry_write_key(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_get_telemetry_write_key = MagicMock()
    monkeypatch.setattr(
        "rasa.telemetry._get_telemetry_write_key", mock_get_telemetry_write_key
    )
    return mock_get_telemetry_write_key


@pytest.fixture
def mock_segment_request_header(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_segment_request_header = MagicMock()
    monkeypatch.setattr(
        "rasa.telemetry.segment_request_header", mock_segment_request_header
    )
    return mock_segment_request_header


@pytest.fixture
def mock_requests_post(monkeypatch: MonkeyPatch) -> MagicMock:
    mock_requests_post = MagicMock()
    monkeypatch.setattr("rasa.telemetry.requests.post", mock_requests_post)
    return mock_requests_post


def test_send_request(
    mock_is_telemetry_debug_enabled: MagicMock,
    mock_print_telemetry_payload: MagicMock,
    mock_get_telemetry_write_key: MagicMock,
    mock_segment_request_header: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    mock_is_telemetry_debug_enabled.return_value = False
    telemetry_key = "some_key"
    mock_get_telemetry_write_key.return_value = telemetry_key
    headers = {"some": "header"}
    mock_segment_request_header.return_value = headers
    mock_response = MagicMock()
    mock_response.json = MagicMock()
    mock_response.json.return_value = {"success": "ok"}
    mock_response.status_code = 200
    mock_requests_post.return_value = mock_response

    url = "some_url"
    payload = {"some": "payload"}
    telemetry._send_request(url, payload)

    mock_is_telemetry_debug_enabled.assert_called_once()
    mock_print_telemetry_payload.assert_not_called()
    mock_get_telemetry_write_key.assert_called_once()
    mock_segment_request_header.assert_called_once_with(telemetry_key)
    mock_requests_post.assert_called_once_with(
        url=url,
        json=payload,
        headers=headers,
        timeout=SEGMENT_REQUEST_TIMEOUT,
    )
    mock_response.json.assert_called_once()


def test_send_request_telemetry_debug_enabled(
    mock_is_telemetry_debug_enabled: MagicMock,
    mock_print_telemetry_payload: MagicMock,
    mock_get_telemetry_write_key: MagicMock,
    mock_segment_request_header: MagicMock,
    mock_requests_post: MagicMock,
) -> None:
    mock_is_telemetry_debug_enabled.return_value = True

    payload = {"some": "payload"}
    telemetry._send_request("some_url", payload)

    mock_is_telemetry_debug_enabled.assert_called_once()
    mock_print_telemetry_payload.assert_called_once_with(payload)
    mock_get_telemetry_write_key.assert_not_called()
    mock_segment_request_header.assert_not_called()
    mock_requests_post.assert_not_called()


def test_send_request_with_invalid_write_key(
    mock_is_telemetry_debug_enabled: MagicMock,
    mock_print_telemetry_payload: MagicMock,
    mock_get_telemetry_write_key: MagicMock,
    mock_segment_request_header: MagicMock,
    mock_requests_post: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    mock_is_telemetry_debug_enabled.return_value = False
    telemetry_key = None
    mock_get_telemetry_write_key.return_value = telemetry_key

    with caplog.at_level(logging.DEBUG):
        telemetry._send_request("some_url", {"some": "payload"})

    mock_is_telemetry_debug_enabled.assert_called_once()
    mock_print_telemetry_payload.assert_not_called()
    mock_get_telemetry_write_key.assert_called_once()
    mock_segment_request_header.assert_not_called()
    mock_requests_post.assert_not_called()

    log_msg = "Skipping request to external service: telemetry key not set."
    assert log_msg in caplog.text


def test_send_request_received_unsuccessful_response(
    mock_is_telemetry_debug_enabled: MagicMock,
    mock_print_telemetry_payload: MagicMock,
    mock_get_telemetry_write_key: MagicMock,
    mock_segment_request_header: MagicMock,
    mock_requests_post: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    mock_is_telemetry_debug_enabled.return_value = False
    telemetry_key = "some_key"
    mock_get_telemetry_write_key.return_value = telemetry_key
    headers = {"some": "header"}
    mock_segment_request_header.return_value = headers
    mock_response = MagicMock()
    mock_response.text = "some error"
    mock_response.status_code = 400
    mock_requests_post.return_value = mock_response

    url = "some_url"
    payload = {"some": "payload"}
    with caplog.at_level(logging.DEBUG):
        telemetry._send_request(url, payload)

    mock_is_telemetry_debug_enabled.assert_called_once()
    mock_print_telemetry_payload.assert_not_called()
    mock_get_telemetry_write_key.assert_called_once()
    mock_segment_request_header.assert_called_once_with(telemetry_key)
    mock_requests_post.assert_called_once_with(
        url=url,
        json=payload,
        headers=headers,
        timeout=SEGMENT_REQUEST_TIMEOUT,
    )

    log_msg = "Segment telemetry request returned a 400 response. Body: some error"
    assert log_msg in caplog.text


def test_send_request_succeeds_without_success_field_in_response(
    mock_is_telemetry_debug_enabled: MagicMock,
    mock_print_telemetry_payload: MagicMock,
    mock_get_telemetry_write_key: MagicMock,
    mock_segment_request_header: MagicMock,
    mock_requests_post: MagicMock,
    caplog: LogCaptureFixture,
) -> None:
    mock_is_telemetry_debug_enabled.return_value = False
    telemetry_key = "some_key"
    mock_get_telemetry_write_key.return_value = telemetry_key
    headers = {"some": "header"}
    mock_segment_request_header.return_value = headers
    mock_response = MagicMock()
    mock_response.json = MagicMock()
    json_data = {"missing_success": "missing"}
    mock_response.json.return_value = json_data
    mock_response.status_code = 200
    mock_requests_post.return_value = mock_response

    url = "some_url"
    payload = {"some": "payload"}
    with caplog.at_level(logging.DEBUG):
        telemetry._send_request(url, payload)

    mock_is_telemetry_debug_enabled.assert_called_once()
    mock_print_telemetry_payload.assert_not_called()
    mock_get_telemetry_write_key.assert_called_once()
    mock_segment_request_header.assert_called_once_with(telemetry_key)
    mock_requests_post.assert_called_once_with(
        url=url,
        json=payload,
        headers=headers,
        timeout=SEGMENT_REQUEST_TIMEOUT,
    )
    mock_response.json.assert_called_once()

    log_msg = f"Segment telemetry request returned a failure. Response: {json_data}"
    assert log_msg in caplog.text


@pytest.mark.parametrize(
    "llm_config,"
    "flow_retrieval_config,"
    "expected_llm_custom_prompt_used,"
    "expected_llm_model_name,"
    "expected_flow_retrieval_enabled,"
    "expected_flow_retrieval_embedding_model_name",
    [
        # default config
        (
            None,
            None,
            False,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model_name"],
            True,
            DEFAULT_EMBEDDINGS_CONFIG["model"],
        ),
        # custom prompt
        (
            {"prompt": "This is custom prompt"},
            None,
            False,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model_name"],
            True,
            DEFAULT_EMBEDDINGS_CONFIG["model"],
        ),
        # turned off flow retrieval
        (
            None,
            {"active": False},
            False,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model_name"],
            False,
            None,
        ),
        # custom llm, custom flow retrieval
        (
            {"model_name": "test_llm"},
            {"embeddings": {"model": "test_embedding"}},
            False,
            "test_llm",
            True,
            "test_embedding",
        ),
    ],
)
def test_get_llm_command_generator_config(
    llm_config: Dict[Text, Any],
    flow_retrieval_config: Dict[Text, Any],
    expected_llm_custom_prompt_used: bool,
    expected_llm_model_name: Text,
    expected_flow_retrieval_enabled: bool,
    expected_flow_retrieval_embedding_model_name: bool,
):
    # Given
    config = """
        recipe: default.v1
        language: en
        pipeline:
        - name: KeywordIntentClassifier
        - name: NLUCommandAdapter
        - name: LLMCommandGenerator
        policies:
        - name: FlowPolicy
        - name: EnterpriseSearchPolicy
        - name: IntentlessPolicy
    """
    config = yaml.load(config, Loader=yaml.FullLoader)
    if llm_config is not None:
        config["pipeline"][2]["llm"] = llm_config
    if flow_retrieval_config is not None:
        config["pipeline"][2]["flow_retrieval"] = flow_retrieval_config

    # When
    result = _get_llm_command_generator_config(config)

    # Then
    assert (
        result[LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED]
        == expected_llm_custom_prompt_used
    )
    assert result[LLM_COMMAND_GENERATOR_MODEL_NAME] == expected_llm_model_name
    assert result[FLOW_RETRIEVAL_ENABLED] == expected_flow_retrieval_enabled
    assert (
        result[FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME]
        == expected_flow_retrieval_embedding_model_name
    )


def test_get_llm_command_generator_config_no_command_generator_component():
    # Given
    config = """
        recipe: default.v1
        language: en
        pipeline:
        - name: KeywordIntentClassifier
    """
    config = yaml.load(config, Loader=yaml.FullLoader)
    # When
    result = _get_llm_command_generator_config(config)
    # Then
    assert result == {
        LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED: None,
        LLM_COMMAND_GENERATOR_MODEL_NAME: None,
        FLOW_RETRIEVAL_ENABLED: None,
        FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME: None,
    }
