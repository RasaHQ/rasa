import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Text
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import responses
import structlog
import yaml
from pytest import LogCaptureFixture, MonkeyPatch

import rasa.api
import rasa.constants
import rasa.utils.licensing
from rasa import telemetry
from rasa.cli.inspect import inspect
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentInfo,
    ProtocolConfig,
)
from rasa.core.brokers.kafka import KafkaEventBroker
from rasa.core.config.available_endpoints import MCPMetaMapConfig, MCPServerConfig
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_LLM_CONFIG as LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG,
)
from rasa.dialogue_understanding.generator.flow_retrieval import (
    DEFAULT_EMBEDDINGS_CONFIG,
)
from rasa.e2e_test.e2e_test_case import (
    Fixture,
    Metadata,
    TestCase,
    TestCaseFixtures,
    TestSuite,
)
from rasa.privacy.privacy_config import PrivacyConfig
from rasa.shared.constants import (
    CONFIG_LANGUAGE_KEY,
    CONFIG_PIPELINE_KEY,
    CONFIG_POLICIES_KEY,
    CONFIG_RECIPE_KEY,
    DEFAULT_INCLUDE_DATE_TIME,
    DEFAULT_TIMEZONE,
    INCLUDE_DATE_TIME_CONFIG_KEY,
    TIMEZONE_CONFIG_KEY,
)
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.flow_step_links import FlowStepLinks, StaticFlowStepLink
from rasa.shared.core.flows.flow_step_sequence import FlowStepSequence
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.telemetry import (
    E2E_TEST_CONVERSION_FILE_TYPE,
    E2E_TEST_CONVERSION_TEST_CASE_COUNT,
    FLOW_RETRIEVAL_EMBEDDING_MODEL_GROUP_ID,
    FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME,
    FLOW_RETRIEVAL_ENABLED,
    LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED,
    LLM_COMMAND_GENERATOR_INCLUDE_DATE_TIME,
    LLM_COMMAND_GENERATOR_MODEL_GROUP_ID,
    LLM_COMMAND_GENERATOR_MODEL_NAME,
    LLM_COMMAND_GENERATOR_TIMEZONE,
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
    TELEMETRY_INSPECT_STARTED_EVENT,
    TELEMETRY_PRIVACY_ENABLED_EVENT,
    TELEMETRY_SERVER_STARTED_EVENT,
    TELEMETRY_UPLOAD_TO_STUDIO_FAILED_EVENT,
    TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE,
    TRACING_BACKEND,
    TRAINING_COMPLETED_EVENT,
    TRAINING_FAILED_EVENT,
    TRAINING_STARTED_EVENT,
    _get_llm_command_generator_config,
)
from rasa.utils import licensing
from rasa.utils.endpoints import read_property_config_from_endpoints_file
from rasa.utils.licensing import LICENSE_ENV_VAR, LICENSE_ENV_VAR_LEGACY

if TYPE_CHECKING:
    from rasa.core.brokers.broker import EventBroker

TELEMETRY_TEST_USER = "083642a3e448423ca652134f00e7fc76"  # just some random static id
TELEMETRY_TEST_KEY = "5640e893c1324090bff26f655456caf3"  # just some random static id
ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA = {
    "vector_store_type": "qdrant",
    "embeddings_type": DEFAULT_EMBEDDINGS_CONFIG["provider"],
    "embeddings_model": DEFAULT_EMBEDDINGS_CONFIG["model"],
    "embeddings_model_group_id": None,
    "llm_type": LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
    "llm_model": LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
    "llm_model_group_id": None,
    "citation_enabled": True,
    "relevancy_check_enabled": True,
    "include_date_time": None,
    "timezone": None,
}


@pytest.fixture(autouse=True)
def patch_global_config_path(tmp_path: Path) -> Generator[None, None, None]:
    """Ensure we use a unique config path for each test to avoid tests influencing
    each other.
    """
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


def _fixtures_per_test_from_list(fixtures: List[Fixture]) -> List[TestCaseFixtures]:
    """Build fixtures_per_test from a list of fixtures (for tests)."""
    return [TestCaseFixtures(test_case_name="", file="", fixtures=fixtures)]


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
        assert (
            b["context"]["license_hash"]
            == hashlib.sha256(valid_license.encode("utf-8")).hexdigest()
        )


def test_get_telemetry_id_valid(monkeypatch: MonkeyPatch, valid_license: Text) -> None:
    monkeypatch.setenv(LICENSE_ENV_VAR, valid_license)

    assert telemetry.get_telemetry_id() is not None


def test_get_telemetry_id_no_license(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv(LICENSE_ENV_VAR, raising=False)
    monkeypatch.delenv(LICENSE_ENV_VAR_LEGACY, raising=False)

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
            TestSuite(
                get_test_cases(),
                _fixtures_per_test_from_list(get_test_fixtures()),
                get_test_metadata(),
                {},
            ),
            3,
            2,
            True,
            True,
            1,
        ),
        (
            TestSuite(
                [],
                _fixtures_per_test_from_list(get_test_fixtures()),
                get_test_metadata(),
                {},
            ),
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
            TestSuite(
                get_test_cases(),
                _fixtures_per_test_from_list(get_test_fixtures()),
                [],
                {},
            ),
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
            TestSuite(
                [],
                _fixtures_per_test_from_list(get_test_fixtures()),
                [],
                {},
            ),
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
        test_suite.test_cases,
        test_suite.fixtures_per_test,
        test_suite.metadata,
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

    with structlog.testing.capture_logs() as caplog:
        telemetry._track(event_name="event", properties={}, context={})

    mock_get_telemetry_id.assert_called_once()
    mock_send_event.assert_not_called()
    mock_with_default_context_fields.assert_not_called()

    log_msg = "Will not report telemetry events as no ID was found."
    assert log_msg in caplog[0]["event_info"]


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
    mock_get_telemetry_write_key.assert_called_once()
    mock_segment_request_header.assert_called_once()
    mock_requests_post.assert_called_once()


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

    with structlog.testing.capture_logs() as caplog:
        telemetry._send_request("some_url", {"some": "payload"})

    mock_is_telemetry_debug_enabled.assert_called_once()
    mock_print_telemetry_payload.assert_not_called()
    mock_get_telemetry_write_key.assert_called_once()
    mock_segment_request_header.assert_not_called()
    mock_requests_post.assert_not_called()

    log_msg = "Skipping request to external service: telemetry key not set."
    assert log_msg in caplog[0]["event_info"]


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
    with structlog.testing.capture_logs() as caplog:
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
    assert log_msg in caplog[0]["event_info"]


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
    with structlog.testing.capture_logs() as caplog:
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
    assert log_msg in caplog[0]["event_info"]


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
    config = f"""
        {CONFIG_RECIPE_KEY}: default.v1
        {CONFIG_LANGUAGE_KEY}: en
        {CONFIG_PIPELINE_KEY}:
        - name: KeywordIntentClassifier
        - name: NLUCommandAdapter
        - name: LLMCommandGenerator
        {CONFIG_POLICIES_KEY}:
        - name: FlowPolicy
        - name: EnterpriseSearchPolicy
        - name: IntentlessPolicy
    """
    config = yaml.load(config, Loader=yaml.FullLoader)
    if llm_config is not None:
        config[CONFIG_PIPELINE_KEY][2]["llm"] = llm_config
    if prompt_config is not None:
        config[CONFIG_PIPELINE_KEY][2]["prompt"] = prompt_config
    if flow_retrieval_config is not None:
        config[CONFIG_PIPELINE_KEY][2]["flow_retrieval"] = flow_retrieval_config

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
    # Check default datetime configuration
    assert result[LLM_COMMAND_GENERATOR_INCLUDE_DATE_TIME] == DEFAULT_INCLUDE_DATE_TIME
    assert result[LLM_COMMAND_GENERATOR_TIMEZONE] == DEFAULT_TIMEZONE


@pytest.mark.parametrize(
    "datetime_config, expected_include_date_time, expected_timezone",
    [
        # default config (no datetime config provided)
        (None, DEFAULT_INCLUDE_DATE_TIME, DEFAULT_TIMEZONE),
        # custom include_date_time only
        ({INCLUDE_DATE_TIME_CONFIG_KEY: False}, False, DEFAULT_TIMEZONE),
        # custom timezone only
        (
            {TIMEZONE_CONFIG_KEY: "America/New_York"},
            DEFAULT_INCLUDE_DATE_TIME,
            "America/New_York",
        ),
        # both custom
        (
            {INCLUDE_DATE_TIME_CONFIG_KEY: True, TIMEZONE_CONFIG_KEY: "Europe/London"},
            True,
            "Europe/London",
        ),
    ],
)
def test_get_llm_command_generator_config_with_datetime_config(
    datetime_config: Optional[Dict[Text, Any]],
    expected_include_date_time: bool,
    expected_timezone: Text,
):
    """Test that datetime configuration is extracted correctly from LLM command generator config."""
    # Given
    config = f"""
        {CONFIG_RECIPE_KEY}: default.v1
        {CONFIG_LANGUAGE_KEY}: en
        {CONFIG_PIPELINE_KEY}:
        - name: KeywordIntentClassifier
        - name: NLUCommandAdapter
        - name: LLMCommandGenerator
        {CONFIG_POLICIES_KEY}:
        - name: FlowPolicy
        - name: EnterpriseSearchPolicy
        - name: IntentlessPolicy
    """
    config = yaml.load(config, Loader=yaml.FullLoader)
    if datetime_config is not None:
        config[CONFIG_PIPELINE_KEY][2].update(datetime_config)

    # When
    result = _get_llm_command_generator_config(config)

    # Then
    assert result[LLM_COMMAND_GENERATOR_INCLUDE_DATE_TIME] == expected_include_date_time
    assert result[LLM_COMMAND_GENERATOR_TIMEZONE] == expected_timezone


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
        config[CONFIG_PIPELINE_KEY][2]["llm"] = llm_config
    if prompt_config is not None:
        config[CONFIG_PIPELINE_KEY][2]["prompt_templates"] = prompt_config
    if flow_retrieval_config is not None:
        config[CONFIG_PIPELINE_KEY][2]["flow_retrieval"] = flow_retrieval_config

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
        LLM_COMMAND_GENERATOR_MODEL_GROUP_ID: None,
        MULTI_STEP_LLM_COMMAND_GENERATOR_HANDLE_FLOWS_PROMPT_USED: None,
        MULTI_STEP_LLM_COMMAND_GENERATOR_FILL_SLOTS_PROMPT_USED: None,
        FLOW_RETRIEVAL_ENABLED: None,
        FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME: None,
        FLOW_RETRIEVAL_EMBEDDING_MODEL_GROUP_ID: None,
        LLM_COMMAND_GENERATOR_INCLUDE_DATE_TIME: None,
        LLM_COMMAND_GENERATOR_TIMEZONE: None,
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
        None,  # model group id for router
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
        None,  # model group id for router
        True,
        True,
    )
    mock_track.assert_called_once_with(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_COMPLETED_EVENT,
        ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA,
    )


@patch("rasa.telemetry._track")
def test_track_enterprise_search_policy_train_completed_with_datetime_config(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that datetime configuration is tracked in enterprise search policy training."""
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    include_date_time = True
    timezone = "America/New_York"

    telemetry.track_enterprise_search_policy_train_completed(
        "qdrant",
        DEFAULT_EMBEDDINGS_CONFIG["provider"],
        DEFAULT_EMBEDDINGS_CONFIG["model"],
        None,
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
        None,
        True,
        True,
        include_date_time=include_date_time,
        timezone=timezone,
    )

    expected_data = ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA.copy()
    expected_data["include_date_time"] = include_date_time
    expected_data["timezone"] = timezone

    mock_track.assert_called_once_with(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_COMPLETED_EVENT,
        expected_data,
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
        None,
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
        None,
        True,
        True,
    )

    mock_track.assert_called_once_with(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_PREDICT_EVENT,
        ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA,
    )


@patch("rasa.telemetry._track")
def test_track_enterprise_search_policy_predict_with_datetime_config(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that datetime configuration is tracked in enterprise search policy prediction."""
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    include_date_time = False
    timezone = "Europe/London"

    telemetry.track_enterprise_search_policy_predict(
        "qdrant",
        DEFAULT_EMBEDDINGS_CONFIG["provider"],
        DEFAULT_EMBEDDINGS_CONFIG["model"],
        None,
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["provider"],
        LLM_COMMAND_GENERATOR_DEFAULT_LLM_CONFIG["model"],
        None,
        True,
        True,
        include_date_time=include_date_time,
        timezone=timezone,
    )

    expected_data = ENTERPRISE_SEARCH_TELEMETRY_EVENT_DATA.copy()
    expected_data["include_date_time"] = include_date_time
    expected_data["timezone"] = timezone

    mock_track.assert_called_once_with(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_PREDICT_EVENT,
        expected_data,
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


@patch("rasa.cli.run.run")
@patch("rasa.telemetry._track")
def test_track_rasa_inspect_telemetry(
    mock_track: MagicMock,
    mock_run: MagicMock,
    monkeypatch: MonkeyPatch,
    inspect_parser: argparse.ArgumentParser,
    endpoints_path: Text,
    trained_rasa_model_with_flows: Text,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    # when rasa inspect is called
    args = inspect_parser.parse_args(
        [
            "inspect",
            "--endpoints",
            endpoints_path,
            "--model",
            trained_rasa_model_with_flows,
        ]
    )
    inspect(args)
    mock_track.assert_called_once_with(
        TELEMETRY_INSPECT_STARTED_EVENT,
        {
            "type": "inspector",
            "assistant_id": "unique_stack_assistant_test_name",
        },
    )
    mock_run.assert_called_once()


@patch("rasa.telemetry._track")
def test_track_upload_to_studio_failed(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    test_response_json = {"error": "some error"}

    telemetry.track_upload_to_studio_failed(
        test_response_json,
    )

    mock_track.assert_called_once_with(
        TELEMETRY_UPLOAD_TO_STUDIO_FAILED_EVENT,
        {
            "studio_response_json": test_response_json,
        },
    )


@patch("rasa.telemetry._track")
def test_train_telemetry_completed(
    mock_track: MagicMock,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
):
    monkeypatch.setattr("rasa.model_training._train_graph", AsyncMock())

    output = str(tmp_path / "models")

    rasa.api.train(
        domain_path,
        stack_config_path,
        [stories_path, nlu_data_path],
        output=output,
    )

    assert mock_track.call_count == 2

    first_call, second_call = mock_track.mock_calls
    assert first_call.args[0] == TRAINING_STARTED_EVENT
    assert first_call.args[1]["assistant_id"] == "unique_stack_assistant_test_name"
    assert second_call.args[0] == TRAINING_COMPLETED_EVENT
    assert second_call.args[1]["assistant_id"] == "unique_stack_assistant_test_name"
    assert second_call.args[1]["training_id"] == first_call.args[1]["training_id"]
    assert "runtime" in second_call.args[1]


@patch("rasa.telemetry._track")
def test_train_telemetry_failed(
    mock_track: MagicMock,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
):
    monkeypatch.setattr(
        "rasa.model_training._train_graph", AsyncMock(side_effect=Exception("Boom"))
    )

    output = str(tmp_path / "models")

    with pytest.raises(Exception):
        rasa.api.train(
            domain_path,
            stack_config_path,
            [stories_path, nlu_data_path],
            output=output,
        )

    assert mock_track.call_count == 2

    first_call, second_call = mock_track.mock_calls
    assert first_call.args[0] == TRAINING_STARTED_EVENT
    assert first_call.args[1]["assistant_id"] == "unique_stack_assistant_test_name"
    assert second_call.args[0] == TRAINING_FAILED_EVENT
    assert second_call.args[1]["assistant_id"] == "unique_stack_assistant_test_name"
    assert second_call.args[1]["training_id"] == first_call.args[1]["training_id"]
    assert "runtime" in second_call.args[1]


@patch("rasa.telemetry._track")
def test_train_telemetry_failed_system_exit(
    mock_track: MagicMock,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
):
    monkeypatch.setattr(
        "rasa.model_training._train_graph", AsyncMock(side_effect=SystemExit(1))
    )

    output = str(tmp_path / "models")

    with pytest.raises(SystemExit):
        rasa.api.train(
            domain_path,
            stack_config_path,
            [stories_path, nlu_data_path],
            output=output,
        )

    assert mock_track.call_count == 2

    first_call, second_call = mock_track.mock_calls
    assert first_call.args[0] == TRAINING_STARTED_EVENT
    assert first_call.args[1]["assistant_id"] == "unique_stack_assistant_test_name"
    assert second_call.args[0] == TRAINING_FAILED_EVENT
    assert second_call.args[1]["assistant_id"] == "unique_stack_assistant_test_name"
    assert second_call.args[1]["training_id"] == first_call.args[1]["training_id"]
    assert "runtime" in second_call.args[1]


@patch("rasa.telemetry._track")
def test_track_server_started(
    mock_track: MagicMock,
    trained_rasa_model_with_flows: Text,
    monkeypatch: MonkeyPatch,
):
    from rasa.core.channels import SlackInput

    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    telemetry.track_server_start(
        [SlackInput], None, trained_rasa_model_with_flows, 4, True
    )

    assert mock_track.call_count == 1
    mock_call = mock_track.mock_calls[0]
    assert mock_call.args[0] == TELEMETRY_SERVER_STARTED_EVENT
    assert mock_call.args[1]["input_channels"] == ["slack"]
    assert mock_call.args[1]["api_enabled"] is True
    assert mock_call.args[1]["number_of_workers"] == 4
    assert mock_call.args[1]["assistant_id"] == "unique_stack_assistant_test_name"
    assert mock_call.args[1]["project"] is not None


@pytest.mark.parametrize(
    "event_broker, expected_stream_pii",
    [
        (None, False),
        (KafkaEventBroker(url="localhost:9092", stream_pii=True), True),
        (KafkaEventBroker(url="localhost:9092", stream_pii=False), False),
        (KafkaEventBroker(url="localhost:9092"), True),
    ],
)
@patch("rasa.telemetry._track")
def test_track_privacy_enabled(
    mock_track: MagicMock,
    monkeypatch: MonkeyPatch,
    event_broker: Optional["EventBroker"],
    expected_stream_pii: bool,
):
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")
    privacy_config_data = read_property_config_from_endpoints_file(
        "data/test_privacy/endpoints_with_valid_privacy.yml", property_name="privacy"
    )
    privacy_config = PrivacyConfig.from_dict(privacy_config_data)
    telemetry.track_privacy_enabled(privacy_config, event_broker)

    assert mock_track.call_count == 1
    mock_call = mock_track.mock_calls[0]
    assert mock_call.args[0] == TELEMETRY_PRIVACY_ENABLED_EVENT
    assert mock_call.args[1]["num_total_rules"] == 2
    assert mock_call.args[1]["redact_count"] == 1
    assert mock_call.args[1]["mask_count"] == 1
    assert mock_call.args[1]["stream_pii"] is expected_stream_pii
    assert mock_call.args[1]["tracker_store_anonymization_enabled"] is True
    assert mock_call.args[1]["tracker_store_deletion_enabled"] is True
    assert (
        mock_call.args[1]["anonymization_cron_trigger"]
        == "cron[month='*', day='*', day_of_week='6', hour='1', minute='30']"
    )
    assert (
        mock_call.args[1]["deletion_cron_trigger"]
        == "cron[month='*', day='*', day_of_week='0', hour='0', minute='30']"
    )


# Tests for agent configuration telemetry
@patch("rasa.telemetry._track")
def test_track_model_training_includes_key_properties(
    mock_track: MagicMock,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
):
    """Test that track_model_training includes agent configuration data."""
    monkeypatch.setattr("rasa.model_training._train_graph", AsyncMock())
    monkeypatch.setenv(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE, "true")

    output = str(tmp_path / "models")

    rasa.api.train(
        domain_path,
        stack_config_path,
        [stories_path, nlu_data_path],
        output=output,
    )

    # Check that both TRAINING_STARTED and TRAINING_COMPLETED events were called
    assert mock_track.call_count == 2

    first_call, second_call = mock_track.mock_calls
    assert first_call.args[0] == TRAINING_STARTED_EVENT

    # Validate agent configuration is present in the tracking data
    tracking_data = first_call.args[1]
    assert "agents" in tracking_data
    assert isinstance(tracking_data["agents"], dict)
    assert isinstance(tracking_data["pipeline"], str)
    assert isinstance(tracking_data["policies"], str)
    assert isinstance(tracking_data["model_groups"], str)
    assert isinstance(tracking_data["recipe"], str)


def test_collect_agent_configuration_empty_flows():
    """Test _collect_agent_configuration with empty flows."""
    # Create empty flows list
    flows = FlowsList([])

    result = telemetry._collect_agent_configuration(flows)

    # Should return empty dict when no flows
    assert result == {}


def test_collect_agent_configuration_no_agents_or_servers():
    """Test _collect_agent_configuration when no agents or MCP servers are available."""
    # Create flows with steps but no agents/servers
    flow = Flow(
        id="test_flow",
        step_sequence=FlowStepSequence(
            [
                CallFlowStep(
                    call="another_flow",
                    custom_id="id",
                    idx=0,
                    description="",
                    metadata={},
                    next=FlowStepLinks([StaticFlowStepLink("flow")]),
                    flow_id="flow",
                )
            ]
        ),
    )
    flows = FlowsList([flow])

    # Mock AvailableAgents and Configuration to return empty
    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.endpoints.mcp_servers = []
        mock_config.return_value.available_agents.agents = {}

        result = telemetry._collect_agent_configuration(flows)

        # Should return empty dict when no agents or servers
        assert result == {}


def test_collect_agent_configuration_with_mcp_tools():
    """Test _collect_agent_configuration with MCP tool calls."""
    # Create flow with MCP tool call

    mcp_step = CallFlowStep(
        call="mcp_tool_name",
        mcp_server="test_server",
        mapping={"param": "value"},
        custom_id="id",
        idx=0,
        description="",
        metadata={},
        next=FlowStepLinks([StaticFlowStepLink("flow")]),
        flow_id="flow",
    )

    flow = Flow(id="test_flow", step_sequence=FlowStepSequence([mcp_step]))
    flows = FlowsList([flow])

    # Mock agents and MCP servers
    mock_agent_info = AgentInfo(
        name="test_agent",
        description="Test agent",
        protocol=ProtocolConfig.RASA,
    )
    mock_mcp_server = MCPServerConfig(
        name="test_server", url="http://localhost:8000", type="http"
    )

    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.available_agents.agents = {
            "test_agent": mock_agent_info
        }
        mock_config.return_value.endpoints.mcp_servers = [mock_mcp_server]

        result = telemetry._collect_agent_configuration(flows)

        # Should include usage data for MCP tool
        assert "usage" in result
        assert len(result["usage"]) == 1
        assert result["usage"][0]["flow"] == "test_flow"
        assert result["usage"][0]["mcp_tool"] == "mcp_tool_name"
        assert result["usage"][0]["mcp_server"] == "test_server"
        assert result["usage"][0]["mapping"] == {"param": "value"}

        # Should include MCP servers and agents
        assert "mcp_servers" in result
        assert "agents" in result
        assert len(result["mcp_servers"]) == 1
        assert len(result["agents"]) == 1


def test_collect_agent_configuration_with_agent_calls():
    """Test _collect_agent_configuration with agent calls."""
    # Create flow with agent call
    agent_step = CallFlowStep(
        call="test_agent",
        exit_if=["some_condition"],
        custom_id="id",
        idx=0,
        description="",
        metadata={},
        next=FlowStepLinks([StaticFlowStepLink("flow")]),
        flow_id="flow",
    )

    flow = Flow(id="test_flow", step_sequence=FlowStepSequence([agent_step]))
    flows = FlowsList([flow])

    # Mock agents and MCP servers
    mock_agent_info = AgentInfo(
        name="test_agent",
        description="Test agent",
        protocol=ProtocolConfig.RASA,
    )
    mock_mcp_server = MCPServerConfig(
        name="test_server", url="http://localhost:8000", type="http"
    )

    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.available_agents.agents = {
            "test_agent": mock_agent_info
        }
        mock_config.return_value.endpoints.mcp_servers = [mock_mcp_server]

        result = telemetry._collect_agent_configuration(flows)

        # Should include usage data for agent call
        assert "usage" in result
        assert len(result["usage"]) == 1
        assert result["usage"][0]["flow"] == "test_flow"
        assert result["usage"][0]["agent"] == "test_agent"
        assert result["usage"][0]["exit_if"] == ["some_condition"]

        # Should include MCP servers and agents
        assert "mcp_servers" in result
        assert "agents" in result


def test_collect_agent_configuration_with_agent_calls_no_exit_if():
    """Test _collect_agent_configuration with agent calls without exit_if."""
    # Create flow with agent call without exit_if
    agent_step = CallFlowStep(
        call="test_agent",
        custom_id="id",
        idx=0,
        description="",
        metadata={},
        next=FlowStepLinks([StaticFlowStepLink("flow")]),
        flow_id="flow",
    )

    flow = Flow(id="test_flow", step_sequence=FlowStepSequence([agent_step]))
    flows = FlowsList([flow])

    # Mock agents and MCP servers
    mock_agent_info = AgentInfo(
        name="test_agent",
        description="Test agent",
        protocol=ProtocolConfig.RASA,
    )
    mock_mcp_server = MCPServerConfig(
        name="test_server", url="http://localhost:8000", type="http"
    )

    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.available_agents.agents = {
            "test_agent": mock_agent_info
        }
        mock_config.return_value.endpoints.mcp_servers = [mock_mcp_server]

        result = telemetry._collect_agent_configuration(flows)

        # Should include usage data for agent call without exit_if
        assert "usage" in result
        assert len(result["usage"]) == 1
        assert result["usage"][0]["flow"] == "test_flow"
        assert result["usage"][0]["agent"] == "test_agent"
        assert "exit_if" not in result["usage"][0]


def test_collect_agent_configuration_skips_flow_calls():
    """Test _collect_agent_configuration skips calls to other flows."""
    # Create flow with call to another flow
    flow_call_step = CallFlowStep(
        call="other_flow",
        custom_id="id",
        idx=0,
        description="",
        metadata={},
        next=FlowStepLinks([StaticFlowStepLink("flow")]),
        flow_id="flow",
    )

    flow = Flow(id="test_flow", step_sequence=FlowStepSequence([flow_call_step]))
    flows = FlowsList([flow])

    # Mock flows to include the called flow
    flows.underlying_flows.append(
        Flow(id="other_flow", step_sequence=FlowStepSequence([]))
    )

    # Mock agents and MCP servers
    mock_agent_info = AgentInfo(
        name="test_agent",
        description="Test agent",
        protocol=ProtocolConfig.A2A,
    )
    mock_mcp_server = MCPServerConfig(
        name="test_server", url="http://localhost:8000", type="http"
    )

    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.available_agents.agents = {
            "test_agent": mock_agent_info
        }
        mock_config.return_value.endpoints.mcp_servers = [mock_mcp_server]

        result = telemetry._collect_agent_configuration(flows)

        # Should not include usage data for flow calls
        assert "usage" in result
        assert len(result["usage"]) == 0


def test_collect_agent_configuration_mcp_servers_serialization():
    """Test that MCP servers are properly serialized."""
    flows = FlowsList([])

    # Create MCP server with some None values
    mock_mcp_server = MCPServerConfig(
        name="test_server",
        url="http://localhost:8000",
        type="http",
        additional_params=None,
    )

    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.available_agents.agents = {}
        mock_config.return_value.endpoints.mcp_servers = [mock_mcp_server]

        result = telemetry._collect_agent_configuration(flows)

        # Should include MCP servers with None values excluded
        assert "mcp_servers" in result
        assert len(result["mcp_servers"]) == 1
        mcp_server_data = result["mcp_servers"][0]
        assert mcp_server_data["name"] == "test_server"
        assert mcp_server_data["url"] == "http://localhost:8000"
        assert mcp_server_data["additional_params"] == {}


def test_collect_agent_configuration_mcp_servers_meta_map_static_keys_only():
    """Telemetry must not include meta_map.static values, only key names."""
    flows = FlowsList([])

    mock_mcp_server = MCPServerConfig(
        name="test_server",
        url="http://localhost:8000",
        type="http",
        meta_map=MCPMetaMapConfig(
            static={"api_version": "v2", "source": "rasa_agent"},
            from_slots=None,
        ),
    )

    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.available_agents.agents = {}
        mock_config.return_value.endpoints.mcp_servers = [mock_mcp_server]

        result = telemetry._collect_agent_configuration(flows)

    mcp_server_data = result["mcp_servers"][0]
    meta_map = mcp_server_data["meta_map"]
    assert meta_map["static_keys"] == ["api_version", "source"]
    assert "static" not in meta_map
    assert "v2" not in json.dumps(mcp_server_data)
    assert "rasa_agent" not in json.dumps(mcp_server_data)


def test_collect_agent_configuration_agents_serialization():
    """Test that agents are properly serialized."""
    flows = FlowsList([])

    # Create agent with some None values
    mock_agent_info = AgentConfig(
        agent=AgentInfo(
            name="test_agent", description="Test agent", protocol=ProtocolConfig.A2A
        ),
        configuration=AgentConfiguration(
            timeout=30,
            max_retries=3,
            agent_card=None,  # This should be excluded
        ),
    )

    with (
        patch(
            "rasa.core.config.configuration.Configuration.get_instance"
        ) as mock_config,
    ):
        mock_config.return_value.available_agents.agents = {
            "test_agent": mock_agent_info
        }
        mock_config.return_value.endpoints.mcp_servers = []

        result = telemetry._collect_agent_configuration(flows)

        # Should include agents with None values excluded
        assert "agents" in result
        assert len(result["agents"]) == 1
        agent_data = result["agents"][0]
        assert "test_agent" in agent_data
        agent_config = agent_data["test_agent"]
        assert agent_config["agent"]["name"] == "test_agent"
        assert agent_config["agent"]["description"] == "Test agent"
        assert agent_config["agent"]["protocol"] == "A2A"
        assert agent_config["configuration"]["timeout"] == 30
        assert agent_config["configuration"]["max_retries"] == 3
        assert "agent_card" not in agent_config["configuration"]  # None values excluded
