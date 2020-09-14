import asyncio
import json
from pathlib import Path
import uuid

from _pytest.monkeypatch import MonkeyPatch
import jsonschema
from mock import Mock
import pytest

from rasa import telemetry
import rasa.constants
from rasa.shared.importers.importer import TrainingDataImporter
from tests.conftest import DEFAULT_CONFIG_PATH

TELEMETRY_TEST_USER = uuid.uuid4().hex
TELEMETRY_TEST_KEY = uuid.uuid4().hex
TELEMETRY_EVENTS_JSON = "docs/docs/telemetry/events.json"


@pytest.fixture(autouse=True)
def patch_global_config_path(monkeypatch: MonkeyPatch, tmp_path: Path):
    """Ensure we use a unique config path for each test to avoid tests influencing
    each other."""
    default_location = rasa.constants.GLOBAL_USER_CONFIG_PATH
    rasa.constants.GLOBAL_USER_CONFIG_PATH = str(tmp_path / "global.yml")
    yield
    rasa.constants.GLOBAL_USER_CONFIG_PATH = default_location


async def test_events_schema(monkeypatch: MonkeyPatch):
    # this allows us to patch the printing part used in debug mode to collect the
    # reported events
    monkeypatch.setenv("RASA_TELEMETRY_DEBUG", "true")
    telemetry.initialize_telemetry()

    mock = Mock()
    monkeypatch.setattr(telemetry, "print_telemetry_event", mock)

    with open(TELEMETRY_EVENTS_JSON) as f:
        schemas = json.load(f)["events"]

    initial = asyncio.Task.all_tasks()
    # Generate all known backend telemetry events, and then use events.json to
    # validate their schema.
    training_data = TrainingDataImporter.load_from_config(DEFAULT_CONFIG_PATH)
    async with telemetry.track_model_training(training_data, "rasa"):
        await asyncio.sleep(1)

    await telemetry.track_telemetry_disabled()

    pending = asyncio.Task.all_tasks() - initial
    await asyncio.gather(*pending)

    assert mock.call_count == 3

    for call in mock.call_args_list:
        event = call.args[0]
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
        "Authorization": "Basic {}".format(
            telemetry.encode_base64(TELEMETRY_TEST_KEY + ":")
        ),
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


async def test_track_ignore_exception(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(telemetry, "_send_event", _mock_track_internal_exception)

    # If the test finishes without raising any exceptions, then it's successful
    assert await telemetry.track("Test") is None


def test_initialize_telemetry():
    telemetry.initialize_telemetry()


def test_initialize_telemetry_with_env_false(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "false")
    assert telemetry.initialize_telemetry() is False


def test_initialize_telemetry_with_env_true(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "true")
    telemetry.initialize_telemetry()
    assert telemetry.initialize_telemetry() is True


def test_initialize_telemetry_env_overwrites_config(monkeypatch: MonkeyPatch):
    telemetry.toggle_telemetry_reporting(True)
    assert telemetry.initialize_telemetry() is True

    monkeypatch.setenv("RASA_TELEMETRY_ENABLED", "false")
    telemetry.initialize_telemetry()
    assert telemetry.initialize_telemetry() is False


def test_initialize_telemetry_prints_info(monkeypatch: MonkeyPatch):
    # Mock actual training
    mock = Mock()
    monkeypatch.setattr(telemetry, "print_telemetry_reporting_info", mock.method)

    telemetry.initialize_telemetry()

    mock.method.assert_called_once()


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


async def test_track_sends_telemetry_id(monkeypatch: MonkeyPatch):
    telemetry.initialize_telemetry()

    mock = Mock()
    monkeypatch.setattr(telemetry, "_send_event", mock)
    await telemetry.track("foobar", {"foo": "bar"}, {"baz": "foo"})

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


def test_environment_write_key_overwrites_key_file(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RASA_TELEMETRY_WRITE_KEY", "foobar")
    assert telemetry.telemetry_write_key() == "foobar"
