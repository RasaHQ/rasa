import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, Text

from _pytest.monkeypatch import MonkeyPatch
import jsonschema
from unittest.mock import MagicMock, Mock
import pytest
import responses

from rasa import telemetry
import rasa.constants
from rasa.core.agent import Agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import CmdlineInput
from rasa.core.tracker_store import TrackerStore
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.shared.nlu.training_data.training_data import TrainingData

TELEMETRY_TEST_USER = "083642a3e448423ca652134f00e7fc76"  # just some random static id
TELEMETRY_TEST_KEY = "5640e893c1324090bff26f655456caf3"  # just some random static id
TELEMETRY_EVENTS_JSON = "docs/docs/telemetry/events.json"


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


async def test_events_schema(
    monkeypatch: MonkeyPatch, default_agent: Agent, config_path: Text
):
    # this allows us to patch the printing part used in debug mode to collect the
    # reported events
    monkeypatch.setenv("RASA_TELEMETRY_DEBUG", "true")
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")

    mock = Mock()
    monkeypatch.setattr(telemetry, "print_telemetry_event", mock)

    with open(TELEMETRY_EVENTS_JSON) as f:
        schemas = json.load(f)["events"]
    initial = asyncio.all_tasks()
    # Generate all known backend telemetry events, and then use events.json to
    # validate their schema.
    training_data = TrainingDataImporter.load_from_config(config_path)

    with telemetry.track_model_training(training_data, "rasa"):
        pass

    telemetry.track_telemetry_disabled()

    telemetry.track_data_split(0.5, "nlu")

    telemetry.track_validate_files(True)

    telemetry.track_data_convert("yaml", "nlu")

    telemetry.track_tracker_export(5, TrackerStore(domain=None), EventBroker())

    telemetry.track_interactive_learning_start(True, False)

    telemetry.track_server_start([CmdlineInput()], None, None, 42, True)

    telemetry.track_project_init("tests/")

    telemetry.track_shell_started("nlu")

    telemetry.track_visualization()

    telemetry.track_core_model_test(5, True, default_agent)

    telemetry.track_nlu_model_test(TrainingData())

    telemetry.track_markers_extraction_initiated("all", False, False, None)

    telemetry.track_markers_extracted(1)

    telemetry.track_markers_stats_computed(1)

    telemetry.track_markers_parsed_count(1, 1, 1)

    # Also track train started for a graph config
    training_data = TrainingDataImporter.load_from_config(
        "data/test_config/graph_config.yml"
    )
    with telemetry.track_model_training(training_data, "rasa"):
        pass

    pending = asyncio.all_tasks() - initial
    await asyncio.gather(*pending)

    assert mock.call_count == 20

    for args, _ in mock.call_args_list:
        event = args[0]
        # `metrics_id` automatically gets added to all event but is
        # not part of the schema so we need to remove it before validation
        del event["properties"]["metrics_id"]
        jsonschema.validate(
            instance=event["properties"], schema=schemas[event["event"]]
        )


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
    monkeypatch.setenv("CI", True)

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
