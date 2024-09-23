import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Text
from unittest.mock import MagicMock, Mock, patch

import pytest
import responses
import yaml
from pytest import LogCaptureFixture, MonkeyPatch

import rasa.api
import rasa.constants
import rasa.utils.licensing
from rasa import telemetry
from rasa.anonymization.anonymisation_rule_yaml_reader import KEY_ANONYMIZATION_RULES
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_LLM_CONFIG as LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG,
)
from rasa.dialogue_understanding.generator.flow_retrieval import (
    DEFAULT_EMBEDDINGS_CONFIG,
)
from rasa.e2e_test.e2e_test_case import Fixture, Metadata, TestCase, TestSuite
from rasa.telemetry import (
    E2E_TEST_CONVERSION_FILE_TYPE,
    E2E_TEST_CONVERSION_TEST_CASE_COUNT,
    FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME,
    FLOW_RETRIEVAL_ENABLED,
    LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED,
    LLM_COMMAND_GENERATOR_MODEL_NAME,
    METRICS_BACKEND,
    MULTI_STEP_LLM_COMMAND_GENERATOR_FILL_SLOTS_PROMPT_USED,
    MULTI_STEP_LLM_COMMAND_GENERATOR_HANDLE_FLOWS_PROMPT_USED,
    SEGMENT_IDENTIFY_ENDPOINT,
    SEGMENT_REQUEST_TIMEOUT,
    SEGMENT_TRACK_ENDPOINT,
    TELEMETRY_E2E_TEST_CONVERSION_EVENT,
    TELEMETRY_E2E_TEST_RUN_STARTED_EVENT,
    TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE,
    TELEMETRY_ENTERPRISE_SEARCH_POLICY_PREDICT_EVENT,
    TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_COMPLETED_EVENT,
    TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_STARTED_EVENT,
    TELEMETRY_ID,
    TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE,
    TRACING_BACKEND,
    _get_llm_command_generator_config,
)
from rasa.utils import licensing
from rasa.utils.licensing import LICENSE_ENV_VAR

TELEMETRY_TEST_USER = "083642a3e448423ca652134f00e7fc76"  # just some random static id
TELEMETRY_TEST_KEY = "5640e893c1324090bff26f655456caf3"  # just some random static id
ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA = {
    "vector_store_type": "qdrant",
    "embeddings_type": DEFAULT_EMBEDDINGS_CONFIG["provider"],
    "embeddings_model": DEFAULT_EMBEDDINGS_CONFIG["model"],
    "llm_type": LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
    "llm_model": LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
    "citation_enabled": True,
}


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


def get_test_cases() -> List[TestCase]:
    return [
        TestCase(name="case 1", steps=[]),
        TestCase(name="case 2", steps=[]),
        TestCase(name="case 3", steps=[]),
    ]


def get_test_fixtures() -> List[Fixture]:
    return [
        Fixture(name="fixture 1", slots_set={}),
        Fixture(name="fixture 2", slots_set={}),
    ]


def get_test_metadata() -> List[Metadata]:
    return [Metadata(name="metadata 1", metadata={})]


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
    assert context["rasa_pro"] == rasa.__version__


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

    monkeypatch.setattr(licensing, "get_license_hash", mock_get_license_hash)

    mock_license_property = MagicMock(return_value=None)
    monkeypatch.setattr(licensing, "property_of_active_license", mock_license_property)

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
                                    "    return rasa.api.train(",
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
    monkeypatch.setattr(licensing, "get_license_hash", lambda: "1234567890")
    monkeypatch.setattr(licensing, "property_of_active_license", lambda _: None)
    context = telemetry._default_context_fields()

    assert "license_hash" in context
    assert context["license_hash"] == "1234567890"

    # make sure it is still there after removing it
    context.pop("license_hash")
    assert "license_hash" in telemetry._default_context_fields()


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
    "tracing_backend, metrics_backend, endpoints_file",
    [
        (
            "otlp",
            "otlp",
            "identify_telemetry_endpoints.yml",
        )
    ],
)
def test_segment_gets_called_for_identify(
    tracing_backend: str,
    metrics_backend: str,
    endpoints_file: str,
    valid_license: str,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)
    monkeypatch.setenv(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE, "foobar")
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")
    endpoints_file = f"data/test_endpoints/{endpoints_file}"
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
        assert b["traits"][METRICS_BACKEND] == metrics_backend
        assert b["traits"][KEY_ANONYMIZATION_RULES] == {
            "enabled": True,
            "metadata": {
                "language": "en",
                "model_provider": "spacy",
                "model_name": "en_core_web_lg",
            },
            "number_of_rule_lists": 1,
            "number_of_rules": 2,
            "substitutions": {"mask": 2, "faker": 0, "text": 0, "not_defined": 0},
            "entities": ["CREDIT_CARD", "IBAN_CODE"],
        }
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
        assert b["traits"][METRICS_BACKEND] is None
        assert b["traits"][KEY_ANONYMIZATION_RULES] == {"enabled": False}
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
    """
    test_suite, expected_number_of_test_cases,
    expected_number_of_fixtures, expected_uses_fixtures,
    expected_uses_metadata, expected_number_of_metadata,
    """,
    [
        (
            TestSuite(get_test_cases(), get_test_fixtures(), get_test_metadata(), {}),
            3,
            2,
            True,
            True,
            1,
        ),
        (
            TestSuite([], get_test_fixtures(), get_test_metadata(), {}),
            0,
            2,
            True,
            True,
            1,
        ),
        (
            TestSuite(get_test_cases(), [], get_test_metadata(), {}),
            3,
            0,
            False,
            True,
            1,
        ),
        (
            TestSuite(get_test_cases(), get_test_fixtures(), [], {}),
            3,
            2,
            True,
            False,
            0,
        ),
        (
            TestSuite(get_test_cases(), [], [], {}),
            3,
            0,
            False,
            False,
            0,
        ),
        (
            TestSuite([], get_test_fixtures(), [], {}),
            0,
            2,
            True,
            False,
            0,
        ),
        (
            TestSuite([], [], get_test_metadata(), {}),
            0,
            0,
            False,
            True,
            1,
        ),
        (
            TestSuite([], [], [], {}),
            0,
            0,
            False,
            False,
            0,
        ),
    ],
)
@patch("rasa.telemetry._track")
def test_track_e2e_test_run(
    mock_track: MagicMock,
    test_suite: TestSuite,
    expected_number_of_test_cases: int,
    expected_number_of_fixtures: int,
    expected_uses_fixtures: bool,
    expected_uses_metadata: bool,
    expected_number_of_metadata: int,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    telemetry.track_e2e_test_run(
        test_suite.test_cases, test_suite.fixtures, test_suite.metadata
    )

    mock_track.assert_called_once_with(
        TELEMETRY_E2E_TEST_RUN_STARTED_EVENT,
        {
            "number_of_test_cases": expected_number_of_test_cases,
            "number_of_fixtures": expected_number_of_fixtures,
            "uses_fixtures": expected_uses_fixtures,
            "uses_metadata": expected_uses_metadata,
            "number_of_metadata": expected_number_of_metadata,
            "uses_assertions": False,
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
    monkeypatch: MonkeyPatch,
) -> None:
    mock_get_telemetry_id.return_value = telemetry_id
    mock_with_default_context_fields.return_value = context
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")
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
    "prompt_config,"
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
            None,
            False,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
            True,
            DEFAULT_EMBEDDINGS_CONFIG["model"],
        ),
        # custom prompt
        (
            None,
            "This is custom prompt",
            None,
            True,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
            True,
            DEFAULT_EMBEDDINGS_CONFIG["model"],
        ),
        # turned off flow retrieval
        (
            None,
            None,
            {"active": False},
            False,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
            False,
            None,
        ),
        # custom llm, custom flow retrieval
        (
            {"model": "test_llm"},
            None,
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
    prompt_config: Text,
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
    if prompt_config is not None:
        config["pipeline"][2]["prompt"] = prompt_config
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


@pytest.mark.parametrize(
    "llm_config,"
    "prompt_config,"
    "flow_retrieval_config,"
    "expected_multi_step_llm_custom_handle_flows_prompt_used,"
    "expected_multi_step_llm_custom_fill_slots_prompt_used,"
    "expected_llm_model_name,"
    "expected_flow_retrieval_enabled,"
    "expected_flow_retrieval_embedding_model_name",
    [
        # default config
        (
            None,
            None,
            None,
            False,
            False,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
            True,
            DEFAULT_EMBEDDINGS_CONFIG["model"],
        ),
        # custom prompt
        (
            None,
            {"fill_slots": "This is custom prompt"},
            None,
            False,
            True,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
            True,
            DEFAULT_EMBEDDINGS_CONFIG["model"],
        ),
        # turned off flow retrieval
        (
            None,
            None,
            {"active": False},
            False,
            False,
            LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
            False,
            None,
        ),
        # custom llm, custom flow retrieval
        (
            {"model": "test_llm"},
            None,
            {"embeddings": {"model": "test_embedding"}},
            False,
            False,
            "test_llm",
            True,
            "test_embedding",
        ),
    ],
)
def test_get_multi_step_llm_command_generator_config(
    llm_config: Dict[Text, Any],
    prompt_config: Dict[Text, Any],
    flow_retrieval_config: Dict[Text, Any],
    expected_multi_step_llm_custom_handle_flows_prompt_used: bool,
    expected_multi_step_llm_custom_fill_slots_prompt_used: bool,
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
        - name: MultiStepLLMCommandGenerator
        policies:
        - name: FlowPolicy
        - name: EnterpriseSearchPolicy
        - name: IntentlessPolicy
    """
    config = yaml.load(config, Loader=yaml.FullLoader)
    if llm_config is not None:
        config["pipeline"][2]["llm"] = llm_config
    if prompt_config is not None:
        config["pipeline"][2]["prompt_templates"] = prompt_config
    if flow_retrieval_config is not None:
        config["pipeline"][2]["flow_retrieval"] = flow_retrieval_config

    # When
    result = _get_llm_command_generator_config(config)

    # Then
    assert (
        result[MULTI_STEP_LLM_COMMAND_GENERATOR_HANDLE_FLOWS_PROMPT_USED]
        == expected_multi_step_llm_custom_handle_flows_prompt_used
    )
    assert (
        result[MULTI_STEP_LLM_COMMAND_GENERATOR_FILL_SLOTS_PROMPT_USED]
        == expected_multi_step_llm_custom_fill_slots_prompt_used
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
        LLM_COMMAND_GENERATOR_MODEL_NAME: None,
        LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED: None,
        MULTI_STEP_LLM_COMMAND_GENERATOR_HANDLE_FLOWS_PROMPT_USED: None,
        MULTI_STEP_LLM_COMMAND_GENERATOR_FILL_SLOTS_PROMPT_USED: None,
        FLOW_RETRIEVAL_ENABLED: None,
        FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME: None,
    }


@patch("rasa.telemetry._track")
def track_track_enterprise_search_policy_train_started(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    telemetry.track_enterprise_search_policy_train_started()
    mock_track.assert_called_once_with(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_STARTED_EVENT
    )


@patch("rasa.telemetry._track")
def test_track_enterprise_search_policy_train_completed(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    telemetry.track_enterprise_search_policy_train_completed(
        "qdrant",
        DEFAULT_EMBEDDINGS_CONFIG["provider"],
        DEFAULT_EMBEDDINGS_CONFIG["model"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
        True,
    )

    mock_track.assert_called_once_with(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_COMPLETED_EVENT,
        ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA,
    )


@patch("rasa.telemetry._track")
def test_track_enterprise_search_policy_predict(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    telemetry.track_enterprise_search_policy_predict(
        "qdrant",
        DEFAULT_EMBEDDINGS_CONFIG["provider"],
        DEFAULT_EMBEDDINGS_CONFIG["model"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
        True,
    )

    mock_track.assert_called_once_with(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_PREDICT_EVENT,
        ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA,
    )


@patch("rasa.telemetry._track")
def test_track_e2e_test_conversion_completed(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    file_type = ".csv"
    test_case_count = 20

    telemetry.track_e2e_test_conversion_completed(
        file_type=file_type,
        test_case_count=test_case_count,
    )

    mock_track.assert_called_once_with(
        TELEMETRY_E2E_TEST_CONVERSION_EVENT,
        {
            E2E_TEST_CONVERSION_FILE_TYPE: file_type,
            E2E_TEST_CONVERSION_TEST_CASE_COUNT: test_case_count,
        },
    )


@patch("rasa.telemetry._track")
def test_track_rasa_train_telemetry_disabled(
    mock_track: MagicMock,
    domain_path: Path,
    stack_config_path: Path,
    stories_path: Text,
    nlu_data_path: Text,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "false")

    # when rasa train is called
    rasa.api.train(
        domain_path,
        stack_config_path,
        [stories_path, nlu_data_path],
        output=str(tmp_path),
    )

    # telemetry should not be tracked
    mock_track.assert_not_called()
