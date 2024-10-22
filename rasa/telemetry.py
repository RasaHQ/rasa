import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import multiprocessing
import os
import platform
import sys
import textwrap
import typing
import uuid
from collections import defaultdict
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Text

import importlib_resources
import requests
from terminaltables import SingleTable

import rasa
import rasa.anonymization.utils
import rasa.shared.utils.io
import rasa.utils.io
from rasa import model
from rasa.constants import (
    CONFIG_FILE_TELEMETRY_KEY,
    CONFIG_TELEMETRY_DATE,
    CONFIG_TELEMETRY_ENABLED,
    CONFIG_TELEMETRY_ID,
)
from rasa.shared.constants import PROMPT_CONFIG_KEY, PROMPT_TEMPLATE_CONFIG_KEY
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.shared.constants import DOCS_URL_TELEMETRY, UTTER_ASK_PREFIX
from rasa.shared.core.flows import Flow
from rasa.shared.core.flows.steps import (
    CollectInformationFlowStep,
    SetSlotsFlowStep,
    LinkFlowStep,
    CallFlowStep,
)
from rasa.shared.exceptions import RasaException
from rasa.utils import common as rasa_utils

if typing.TYPE_CHECKING:
    from rasa.core.brokers.broker import EventBroker
    from rasa.core.tracker_store import TrackerStore
    from rasa.core.channels.channel import InputChannel
    from rasa.core.agent import Agent
    from rasa.shared.nlu.training_data.training_data import TrainingData
    from rasa.shared.importers.importer import TrainingDataImporter
    from rasa.core.utils import AvailableEndpoints
    from rasa.e2e_test.e2e_test_case import TestCase, Fixture, Metadata

logger = logging.getLogger(__name__)

SEGMENT_TRACK_ENDPOINT = "https://api.segment.io/v1/track"
SEGMENT_IDENTIFY_ENDPOINT = "https://api.segment.io/v1/identify"
SEGMENT_REQUEST_TIMEOUT = 5  # seconds

TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE = "RASA_TELEMETRY_ENABLED"
TELEMETRY_DEBUG_ENVIRONMENT_VARIABLE = "RASA_TELEMETRY_DEBUG"
RASA_PRO_CONFIG_FILE_TELEMETRY_KEY = "traits"

# the environment variable can be used for local development to set a test key
# e.g. `RASA_TELEMETRY_WRITE_KEY=12354 rasa train`
TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE = "RASA_TELEMETRY_WRITE_KEY"
EXCEPTION_WRITE_KEY_ENVIRONMENT_VARIABLE = "RASA_EXCEPTION_WRITE_KEY"

TELEMETRY_ID = "metrics_id"
TELEMETRY_ENABLED_BY_DEFAULT = True

# if one of these environment variables is set, we assume to be running in CI env
CI_ENVIRONMENT_TELL = [
    "bamboo.buildKey",
    "BUILD_ID",
    "BUILD_NUMBER",
    "BUILDKITE",
    "CI",
    "CIRCLECI",
    "CONTINUOUS_INTEGRATION",
    "GITHUB_ACTIONS",
    "HUDSON_URL",
    "JENKINS_URL",
    "TEAMCITY_VERSION",
    "TRAVIS",
    "CODEBUILD_BUILD_ARN",
    "CODEBUILD_BUILD_ID",
    "CODEBUILD_BATCH_BUILD_IDENTIFIER",
]

# If updating or creating a new event, remember to update
# https://rasa.com/docs/rasa-pro/telemetry/telemetry OR
# https://rasa.com/docs/rasa-pro/telemetry/reference
TRAINING_STARTED_EVENT = "Training Started"
TRAINING_COMPLETED_EVENT = "Training Completed"
TELEMETRY_DISABLED_EVENT = "Telemetry Disabled"
TELEMETRY_DATA_SPLIT_EVENT = "Training Data Split"
TELEMETRY_DATA_VALIDATED_EVENT = "Training Data Validated"
TELEMETRY_DATA_CONVERTED_EVENT = "Training Data Converted"
TELEMETRY_TRACKER_EXPORTED_EVENT = "Tracker Exported"
TELEMETRY_INTERACTIVE_LEARNING_STARTED_EVENT = "Interactive Learning Started"
TELEMETRY_SERVER_STARTED_EVENT = "Server Started"
TELEMETRY_PROJECT_CREATED_EVENT = "Project Created"
TELEMETRY_SHELL_STARTED_EVENT = "Shell Started"
TELEMETRY_VISUALIZATION_STARTED_EVENT = "Story Visualization Started"
TELEMETRY_TEST_CORE_EVENT = "Model Core Tested"
TELEMETRY_TEST_NLU_EVENT = "Model NLU Tested"
TELEMETRY_MARKERS_EXTRACTION_INITIATED_EVENT = "Markers Extraction Initiated"
TELEMETRY_MARKERS_EXTRACTED_EVENT = "Markers Extracted"
TELEMETRY_MARKERS_STATS_COMPUTED_EVENT = "Markers Statistics Computed"
TELEMETRY_MARKERS_PARSED_COUNT = "Markers Parsed"

TELEMETRY_RESPONSE_REPHRASED_EVENT = "Response Rephrased"
TELEMETRY_INTENTLESS_POLICY_TRAINING_STARTED_EVENT = (
    "Intentless Policy Training Started"
)
TELEMETRY_INTENTLESS_POLICY_TRAINING_COMPLETED_EVENT = (
    "Intentless Policy Training Completed"
)
TELEMETRY_INTENTLESS_POLICY_PREDICT_EVENT = "Intentless Policy Predicted"
TELEMETRY_E2E_TEST_RUN_STARTED_EVENT = "E2E Test Run Started"
TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_STARTED_EVENT = (
    "Enterprise Search Policy Training Started"
)
TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_COMPLETED_EVENT = (
    "Enterprise Search Policy Training Completed"
)
TELEMETRY_ENTERPRISE_SEARCH_POLICY_PREDICT_EVENT = "Enterprise Search Policy Predicted"

# licensing events
TELEMETRY_CONVERSATION_COUNT = "Conversation Count"
TELEMETRY_CONVERSATION_SOFT_LIMIT_REACHED = "Conversation Soft Limit Reached"
TELEMETRY_CONVERSATION_HARD_LIMIT_REACHED = "Conversation Hard Limit Reached"

# used to calculate the context on the first call and cache it afterwards
TELEMETRY_CONTEXT = None

# constants used for the training started events
NUM_FLOWS = "num_flows"
NUM_FLOWS_WITH_NLU_TRIGGER = "num_flows_with_nlu_trigger"
NUM_FLOWS_WITH_FLOW_GUARDS = "num_flows_with_flow_guards"
NUM_FLOWS_ALWAYS_INCLUDED_IN_PROMPT = "num_flows_always_included_in_prompt"
NUM_FLOWS_WITH_NOT_STARTABLE_FLOW_GUARDS = "num_flows_with_not_startable_flow_guards"
NUM_COLLECT_STEPS = "num_collect_steps"
NUM_COLLECT_STEPS_WITH_SEPARATE_UTTER = "num_collect_steps_with_separate_utter"
NUM_COLLECT_STEPS_WITH_REJECTIONS = "num_collect_steps_with_rejections"
NUM_COLLECT_STEPS_WITH_NOT_RESET_AFTER_FLOW_ENDS = (
    "num_collect_steps_with_not_reset_after_flow_ends"
)
NUM_SET_SLOT_STEPS = "num_set_slot_steps"
MAX_DEPTH_OF_IF_CONSTRUCT = "max_depth_of_if_construct"
NUM_LINK_STEPS = "num_link_steps"
NUM_CALL_STEPS = "num_call_steps"
NUM_SHARED_SLOTS_BETWEEN_FLOWS = "num_shared_slots_between_flows"
LLM_COMMAND_GENERATOR_MODEL_NAME = "llm_command_generator_model_name"
LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED = "llm_command_generator_custom_prompt_used"
MULTI_STEP_LLM_COMMAND_GENERATOR_HANDLE_FLOWS_PROMPT_USED = (
    "multi_step_llm_command_generator_custom_handle_flows_prompt_used"
)
MULTI_STEP_LLM_COMMAND_GENERATOR_FILL_SLOTS_PROMPT_USED = (
    "multi_step_llm_command_generator_custom_fill_slots_prompt_used"
)
FLOW_RETRIEVAL_ENABLED = "flow_retrieval_enabled"
FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME = "flow_retrieval_embedding_model_name"
TRACING_BACKEND = "tracing_backend"
METRICS_BACKEND = "metrics_backend"
VERSION = "version"

# E2E test conversion
TELEMETRY_E2E_TEST_CONVERSION_EVENT = "E2E Test Conversion Completed"
E2E_TEST_CONVERSION_FILE_TYPE = "file_type"
E2E_TEST_CONVERSION_TEST_CASE_COUNT = "test_case_count"


def print_telemetry_reporting_info() -> None:
    """Print telemetry information to std out."""
    message = textwrap.dedent(
        f"""
        Rasa Pro reports anonymous usage telemetry to help improve the product
        for all its users.

        If you'd like to opt-out, you can use `rasa telemetry disable`.
        To learn more, check out {DOCS_URL_TELEMETRY}."""
    ).strip()

    table = SingleTable([[message]])
    print(table.table)


def _default_telemetry_configuration(is_enabled: bool) -> Dict[Text, Any]:
    return {
        CONFIG_TELEMETRY_ENABLED: is_enabled,
        CONFIG_TELEMETRY_ID: uuid.uuid4().hex,
        CONFIG_TELEMETRY_DATE: datetime.now(),
    }


def _write_default_telemetry_configuration(
    is_enabled: bool = TELEMETRY_ENABLED_BY_DEFAULT,
) -> bool:
    new_config = _default_telemetry_configuration(is_enabled)

    keys = [CONFIG_FILE_TELEMETRY_KEY, RASA_PRO_CONFIG_FILE_TELEMETRY_KEY]

    success = all(
        [rasa_utils.write_global_config_value(key, new_config) for key in keys]
    )

    # Do not show info if user has enabled/disabled telemetry via env var
    telemetry_environ = os.environ.get(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE)
    if is_enabled and success and telemetry_environ is None:
        print_telemetry_reporting_info()

    return success


def _is_telemetry_enabled_in_configuration() -> bool:
    """Read telemetry configuration from the user's Rasa config file in $HOME.

    Creates a default configuration if no configuration exists.

    Returns:
        `True`, if telemetry is enabled, `False` otherwise.
    """
    try:
        stored_config = rasa_utils.read_global_config_value(
            CONFIG_FILE_TELEMETRY_KEY, unavailable_ok=False
        )

        return stored_config[CONFIG_TELEMETRY_ENABLED]
    except ValueError as e:
        logger.debug(f"Could not read telemetry settings from configuration file: {e}")

        # seems like there is no config, we'll create one and enable telemetry
        success = _write_default_telemetry_configuration()
        # if writing the configuration failed, telemetry will be disabled
        return TELEMETRY_ENABLED_BY_DEFAULT and success


def is_telemetry_enabled() -> bool:
    """Check if telemetry is enabled either in configuration or environment.

    Returns:
        `True`, if telemetry is enabled, `False` otherwise.
    """
    from rasa.utils import licensing

    if licensing.is_champion_server_license():
        logger.debug("Telemetry is enabled for developer licenses.")
        return True

    telemetry_environ = os.environ.get(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE)

    if telemetry_environ is not None:
        return telemetry_environ.lower() == "true"

    try:
        return rasa_utils.read_global_config_value(
            CONFIG_FILE_TELEMETRY_KEY, unavailable_ok=False
        )[CONFIG_TELEMETRY_ENABLED]
    except ValueError:
        return False


def initialize_telemetry() -> bool:
    """Read telemetry configuration from the user's Rasa config file in $HOME.

    Creates a default configuration if no configuration exists.

    Returns:
        `True`, if telemetry is enabled, `False` otherwise.
    """
    try:
        # calling this even if the environment variable is set makes sure the
        # configuration is created and there is a telemetry ID
        is_enabled_in_configuration = _is_telemetry_enabled_in_configuration()

        telemetry_environ = os.environ.get(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE)

        if telemetry_environ is None:
            return is_enabled_in_configuration

        return telemetry_environ.lower() == "true"
    except Exception as e:  # skipcq:PYL-W0703
        logger.exception(
            f"Failed to initialize telemetry reporting: {e}."
            f"Telemetry reporting will be disabled."
        )
        return False


def ensure_telemetry_enabled(f: Callable[..., Any]) -> Callable[..., Any]:
    """Function decorator for telemetry functions that ensures telemetry is enabled.

    WARNING: does not work as a decorator for async generators.

    Args:
        f: function to call if telemetry is enabled
    Returns:
        Return wrapped function
    """
    # allows us to use the decorator for async generator functions
    if inspect.isasyncgenfunction(f):

        @wraps(f)
        async def decorated_async_gen(*args: Any, **kwargs: Any) -> Any:
            if is_telemetry_enabled():
                yield f(*args, **kwargs)

        return decorated_async_gen

    # allows us to use the decorator for async and non async functions
    if asyncio.iscoroutinefunction(f):

        @wraps(f)
        async def decorated_coroutine(*args: Any, **kwargs: Any) -> Any:
            if is_telemetry_enabled():
                return await f(*args, **kwargs)
            return None

        return decorated_coroutine

    @wraps(f)
    def decorated(*args: Any, **kwargs: Any) -> Any:
        if is_telemetry_enabled():
            return f(*args, **kwargs)
        return None

    return decorated


def _fetch_write_key(tool: Text, environment_variable: Text) -> Optional[Text]:
    """Read the write key from a tool from our set of keys.

    Args:
        tool: name of the tool we want to fetch a key for
        environment_variable: name of the environment variable to set the key
    Returns:
        write key, if a key was present.
    """
    import importlib_resources
    from rasa import __name__ as name

    if os.environ.get(environment_variable):
        # a write key set using the environment variable will always
        # overwrite any key provided as part of the package (`keys` file)
        return os.environ.get(environment_variable)

    write_key_path = str(importlib_resources.files(name).joinpath("keys"))

    # noinspection PyBroadException
    try:
        with open(write_key_path) as f:
            return json.load(f).get(tool)
    except Exception:  # skipcq:PYL-W0703
        return None


def telemetry_write_key() -> Optional[Text]:
    """Read the Segment write key from the segment key text file.

    The segment key text file should by present only in wheel/sdist packaged
    versions of Rasa Pro. This avoids running telemetry locally when
    developing on Rasa or when running CI builds.

    In local development, this should always return `None` to avoid logging telemetry.

    Returns:
        Segment write key, if the key file was present.
    """
    return _fetch_write_key("segment", TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE)


def sentry_write_key() -> Optional[Text]:
    """Read the sentry write key from the sentry key text file.

    Returns:
        Sentry write key, if the key file was present.
    """
    return _fetch_write_key("sentry", EXCEPTION_WRITE_KEY_ENVIRONMENT_VARIABLE)


def _encode_base64(original: Text, encoding: Text = "utf-8") -> Text:
    """Encodes a string as a base64 string.

    Args:
        original: Text to be encoded.
        encoding: Encoding used to convert text to binary.

    Returns:
        Encoded text.
    """
    import base64

    return base64.b64encode(original.encode(encoding)).decode(encoding)


def segment_request_header(write_key: Text) -> Dict[Text, Any]:
    """Use a segment write key to create authentication headers for the segment API.

    Args:
        write_key: Authentication key for segment.

    Returns:
        Authentication headers for segment.
    """
    return {
        "Authorization": "Basic {}".format(_encode_base64(write_key + ":")),
        "Content-Type": "application/json",
    }


def segment_request_payload(
    distinct_id: Text,
    event_name: Text,
    properties: Dict[Text, Any],
    context: Dict[Text, Any],
) -> Dict[Text, Any]:
    """Compose a valid payload for the segment API.

    Args:
        distinct_id: Unique telemetry ID.
        event_name: Name of the event.
        properties: Values to report along the event.
        context: Context information about the event.

    Returns:
        Valid segment payload.
    """
    return {
        "userId": distinct_id,
        "event": event_name,
        "properties": properties,
        "context": context,
    }


def in_continuous_integration() -> bool:
    """Returns `True` if currently running inside a continuous integration context."""
    return any(env in os.environ for env in CI_ENVIRONMENT_TELL)


def _is_telemetry_debug_enabled() -> bool:
    """Check if telemetry debug mode is enabled."""
    return (
        os.environ.get(TELEMETRY_DEBUG_ENVIRONMENT_VARIABLE, "false").lower() == "true"
    )


def print_telemetry_payload(payload: Dict[Text, Any]) -> None:
    """Print a telemetry payload to the commandline.

    Args:
        payload: payload to be delivered to segment.
    """
    payload_json = json.dumps(payload, indent=2)
    logger.debug(f"Telemetry payload: {payload_json}")


def _get_telemetry_write_key() -> Optional[Text]:
    if os.environ.get(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE):
        return os.environ.get(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE)

    write_key_path = str(importlib_resources.files(rasa.__name__).joinpath("keys"))

    try:
        with open(write_key_path) as f:
            return json.load(f).get("segment")
    except Exception:
        return None


def _send_event(
    distinct_id: Text,
    event_name: Text,
    properties: Dict[Text, Any],
    context: Dict[Text, Any],
) -> None:
    """Report the contents segmentof an event to the /track Segment endpoint.

    Documentation: https://.com/docs/sources/server/http/

    Do not call this function from outside telemetry.py! This function does not
    check if telemetry is enabled or not.

    Args:
        distinct_id: Unique telemetry ID.
        event_name: Name of the event.
        properties: Values to report along the event.
        context: Context information about the event.
    """
    payload = segment_request_payload(distinct_id, event_name, properties, context)

    _send_request(SEGMENT_TRACK_ENDPOINT, payload)


def _send_request(url: Text, payload: Dict[Text, Any]) -> None:
    """Send a request to the Segment API.

    Args:
        url: URL of the Segment API endpoint
        payload: payload to send to the Segment API
    """
    if _is_telemetry_debug_enabled():
        print_telemetry_payload(payload)
        return

    write_key = _get_telemetry_write_key()
    if not write_key:
        # If RASA_TELEMETRY_WRITE_KEY is empty or `None`, telemetry has not
        # been enabled for this build (e.g. because it is running from source)
        logger.debug("Skipping request to external service: telemetry key not set.")
        return

    headers = rasa.telemetry.segment_request_header(write_key)

    resp = requests.post(
        url=url,
        headers=headers,
        json=payload,
        timeout=SEGMENT_REQUEST_TIMEOUT,
    )
    # handle different failure cases
    if resp.status_code != 200:
        logger.debug(
            f"Segment telemetry request returned a {resp.status_code} response. "
            f"Body: {resp.text}"
        )
    else:
        data = resp.json()
        if not data.get("success"):
            logger.debug(
                f"Segment telemetry request returned a failure. Response: {data}"
            )


def _hash_directory_path(path: Text) -> Optional[Text]:
    """Create a hash for the directory.

    Returns:
        hash of the directories path
    """
    full_path = Path(path).absolute()
    return hashlib.sha256(str(full_path).encode("utf-8")).hexdigest()


# noinspection PyBroadException
def _is_docker() -> bool:
    """Guess if we are running in docker environment.

    Returns:
        `True` if we are running inside docker, `False` otherwise.
    """
    # first we try to use the env
    try:
        os.stat("/.dockerenv")
        return True
    except Exception:  # skipcq:PYL-W0703
        pass

    # if that didn't work, try to use proc information
    try:
        return "docker" in rasa.shared.utils.io.read_file("/proc/self/cgroup", "utf8")
    except Exception:  # skipcq:PYL-W0703
        return False


def with_default_context_fields(
    context: Optional[Dict[Text, Any]] = None,
) -> Dict[Text, Any]:
    """Return a new context dictionary with default and provided field values merged.

    The default fields contain only the OS information for now.

    Args:
        context: Context information about the event.

    Return:
        A new context.
    """
    context = context or {}

    return {**_default_context_fields(), **context}


def _default_context_fields() -> Dict[Text, Any]:
    """Return a dictionary that contains the default context values.

    Return:
        A new context containing information about the runtime environment.
    """
    from rasa.utils.licensing import property_of_active_license, get_license_hash

    global TELEMETRY_CONTEXT

    if not TELEMETRY_CONTEXT:
        # Make sure to update the example in docs/docs/telemetry/telemetry.mdx
        # if you change / add context
        TELEMETRY_CONTEXT = {
            "os": {"name": platform.system(), "version": platform.release()},
            "ci": in_continuous_integration(),
            "project": model.project_fingerprint(),
            "directory": _hash_directory_path(os.getcwd()),
            "python": sys.version.split(" ")[0],
            "rasa_pro": rasa.__version__,
            "cpu": multiprocessing.cpu_count(),
            "docker": _is_docker(),
            "license_hash": get_license_hash(),
            "company": property_of_active_license(
                lambda active_license: active_license.company
            ),
        }

    # avoid returning the cached dict --> caller could modify the dictionary...
    # usually we would use `lru_cache`, but that doesn't return a dict copy and
    # doesn't work on inner functions, so we need to roll our own caching...
    return TELEMETRY_CONTEXT.copy()


def _track(
    event_name: Text,
    properties: Optional[Dict[Text, Any]] = None,
    context: Optional[Dict[Text, Any]] = None,
) -> None:
    """Tracks a telemetry event.

    It is OK to use this function from outside telemetry.py, but note that it
    is recommended to create a new track_xyz() function for complex telemetry
    events, or events that are generated from many parts of the Rasa Pro code.

    Args:
        event_name: Name of the event.
        properties: Dictionary containing the event's properties.
        context: Dictionary containing some context for this event.
    """
    try:
        telemetry_id = get_telemetry_id()

        if not telemetry_id:
            logger.debug("Will not report telemetry events as no ID was found.")
            return

        if not properties:
            properties = {}

        properties[TELEMETRY_ID] = telemetry_id

        # this is an additional check in case _track() is called
        # from a function that is not decorated with @ensure_telemetry_enabled
        if is_telemetry_enabled():
            _send_event(
                telemetry_id,
                event_name,
                properties,
                with_default_context_fields(context),
            )
    except Exception as e:  # skipcq:PYL-W0703
        logger.debug(f"Skipping telemetry reporting: {e}")


def _identify(
    traits: Optional[Dict[Text, Any]] = None,
    context: Optional[Dict[Text, Any]] = None,
) -> None:
    """Tracks telemetry traits.

    It is OK to use this function from outside telemetry.py, but note that it
    is recommended to create a new track_xyz() function for complex telemetry
    traits, or traits that are generated from many parts of the Rasa Pro code.

    Args:
        traits: Dictionary containing the tracked traits.
        context: Dictionary containing some context for the traits.
    """
    try:
        telemetry_id = get_telemetry_id()

        if not telemetry_id:
            logger.debug("Will not report telemetry events as no ID was found.")
            return

        if not traits:
            traits = {}

        _send_traits(telemetry_id, traits, with_default_context_fields(context))
    except Exception as e:
        logger.debug(f"Skipping telemetry reporting: {e}")


def _send_traits(
    distinct_id: Text,
    traits: Dict[Text, Any],
    context: Dict[Text, Any],
) -> None:
    """Report the contents of telemetry traits to the /identify Segment endpoint.

    Do not call this function from outside telemetry.py! This function does not
    check if telemetry is enabled or not.

    Args:
        distinct_id: Unique telemetry ID.
        traits: Pieces of information to be recorded about
                rasa_plus interface implementations.
        context: Context information to be sent along with traits.
    """
    payload = segment_identify_request_payload(distinct_id, traits, context)

    _send_request(SEGMENT_IDENTIFY_ENDPOINT, payload)


def segment_identify_request_payload(
    distinct_id: Text,
    traits: Dict[Text, Any],
    context: Dict[Text, Any],
) -> Dict[Text, Any]:
    """Compose a valid payload for the segment API.

    Args:
        distinct_id: Unique telemetry ID.
        traits: Pieces of information to be recorded about
                rasa_plus interface implementations.
        context: Context information to be sent along with traits.

    Returns:
        Valid segment payload.
    """
    return {
        "userId": distinct_id,
        "traits": traits,
        "context": context,
    }


def get_telemetry_id() -> Optional[Text]:
    """Return the unique telemetry identifier for this Rasa Pro install.

    The identifier can be based on the license.
    Otherwise, it can be any string, but it should be a UUID.

    Returns:
        The identifier, if it is configured correctly.
    """
    from rasa.utils.licensing import property_of_active_license

    return property_of_active_license(lambda active_license: active_license.jti)


def toggle_telemetry_reporting(is_enabled: bool) -> None:
    """Write to the configuration if telemetry tracking should be enabled or disabled.

    Args:
        is_enabled: `True` if the telemetry reporting should be enabled,
            `False` otherwise.
    """
    configuration = rasa_utils.read_global_config_value(CONFIG_FILE_TELEMETRY_KEY)

    if configuration:
        configuration[CONFIG_TELEMETRY_ENABLED] = is_enabled
    else:
        configuration = _default_telemetry_configuration(is_enabled)

    rasa_utils.write_global_config_value(CONFIG_FILE_TELEMETRY_KEY, configuration)
    rasa_utils.write_global_config_value(
        RASA_PRO_CONFIG_FILE_TELEMETRY_KEY, configuration
    )


def filter_errors(
    event: Optional[Dict[Text, Any]], hint: Optional[Dict[Text, Any]] = None
) -> Optional[Dict[Text, Any]]:
    """Filter errors.

    Args:
        event: event to be logged to sentry
        hint: some hinting information sent alongside of the event

    Returns:
        the event without any sensitive / PII data or `None` if the event constitutes
        an `ImportError` which should be discarded.
    """
    if hint and "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]
        if isinstance(exc_value, ImportError):
            return None
    return event


def before_send(
    event: Dict[Text, Any], _unused_hint: Optional[Dict[Text, Any]] = None
) -> Optional[Dict[Text, Any]]:
    """Strips the sensitive data and filters errors before sending to sentry.

    Args:
        event: event to be logged to sentry
        _unused_hint: some hinting information sent alongside of the event

    Returns:
        the event without any sensitive / PII data or `None` if the event should
        be discarded.
    """
    cleaned_event = strip_sensitive_data_from_sentry_event(event, _unused_hint)
    return filter_errors(cleaned_event, _unused_hint)


def strip_sensitive_data_from_sentry_event(
    event: Dict[Text, Any], _unused_hint: Optional[Dict[Text, Any]] = None
) -> Optional[Dict[Text, Any]]:
    """Remove any sensitive data from the event (e.g. path names).

    Args:
        event: event to be logged to sentry
        _unused_hint: some hinting information sent alongside of the event

    Returns:
        the event without any sensitive / PII data or `None` if the event should
        be discarded.
    """
    # removes any paths from stack traces (avoids e.g. sending
    # a users home directory name if package is installed there)
    for value in event.get("exception", {}).get("values", []):
        for frame in value.get("stacktrace", {}).get("frames", []):
            frame["abs_path"] = ""

            if f"rasa_sdk{os.path.sep}executor.py" in frame["filename"]:
                # this looks a lot like an exception in the SDK and hence custom code
                # no need for us to deal with that
                return None
            elif "site-packages" in frame["filename"]:
                # drop site-packages and following slash / backslash
                relative_name = frame["filename"].split("site-packages")[-1][1:]
                frame["filename"] = os.path.join("site-packages", relative_name)
            elif "dist-packages" in frame["filename"]:
                # drop dist-packages and following slash / backslash
                relative_name = frame["filename"].split("dist-packages")[-1][1:]
                frame["filename"] = os.path.join("dist-packages", relative_name)
            elif os.path.isabs(frame["filename"]):
                # if the file path is absolute, we'll drop the whole event as this is
                # very likely custom code. needs to happen after cleaning as
                # site-packages / dist-packages paths are also absolute, but fine.
                return None
    return event


@ensure_telemetry_enabled
def initialize_error_reporting() -> None:
    """Sets up automated error reporting.

    Exceptions are reported to sentry. We avoid sending any metadata (local
    variables, paths, ...) to make sure we don't compromise any data. Only the
    exception and its stacktrace is logged and only if the exception origins
    from the `rasa` package.
    """
    import sentry_sdk
    from sentry_sdk import configure_scope
    from sentry_sdk.integrations.atexit import AtexitIntegration
    from sentry_sdk.integrations.dedupe import DedupeIntegration
    from sentry_sdk.integrations.excepthook import ExcepthookIntegration

    # key for local testing can be found at
    # https://sentry.io/settings/rasahq/projects/rasa-open-source/install/python/
    # for local testing, set the key using `RASA_EXCEPTION_WRITE_KEY=key rasa <command>`
    key = sentry_write_key()

    if not key:
        return

    telemetry_id = get_telemetry_id()

    # this is a very defensive configuration, avoiding as many integrations as
    # possible. it also submits very little data (exception with error message
    # and line numbers).
    sentry_sdk.init(
        f"https://{key}.ingest.sentry.io/2801673",
        before_send=before_send,
        integrations=[
            ExcepthookIntegration(),
            DedupeIntegration(),
            AtexitIntegration(lambda _, __: None),
        ],
        send_default_pii=False,  # activate PII filter
        server_name=telemetry_id or "UNKNOWN",
        ignore_errors=[
            # std lib errors
            KeyboardInterrupt,  # user hit the interrupt key (Ctrl+C)
            MemoryError,  # machine is running out of memory
            NotImplementedError,  # user is using a feature that is not implemented
            asyncio.CancelledError,  # an async operation has been cancelled by the user
            # expected Rasa errors
            RasaException,
            OSError,
        ],
        in_app_include=["rasa"],  # only submit errors in this package
        with_locals=False,  # don't submit local variables
        release=f"rasa-{rasa.__version__}",
        default_integrations=False,
        environment="development" if in_continuous_integration() else "production",
    )

    if not telemetry_id:
        return

    with configure_scope() as scope:
        # sentry added these more recently, just a protection in a case where a
        # user has installed an older version of sentry
        if hasattr(scope, "set_user"):
            scope.set_user({"id": telemetry_id})

        default_context = _default_context_fields()
        if hasattr(scope, "set_context"):
            if "os" in default_context:
                # os is a nested dict, hence we report it separately
                scope.set_context("Operating System", default_context.pop("os"))
            scope.set_context("Environment", default_context)


@contextlib.contextmanager
def track_model_training(
    training_data: "TrainingDataImporter", model_type: Text, is_finetuning: bool = False
) -> typing.Generator[None, None, None]:
    """Track a model training started.

    WARNING: since this is a generator, it can't use the ensure telemetry
        decorator. We need to manually add these checks here. This can be
        fixed as soon as we drop python 3.6 support.

    Args:
        training_data: Training data used for the training.
        model_type: Specifies the type of training, should be either "rasa", "core"
            or "nlu".
        is_finetuning: `True` if the model is trained by finetuning another model.
    """
    if not initialize_telemetry():
        # telemetry reporting is disabled. we won't do any reporting
        yield  # runs the training
        return

    config = training_data.get_config()
    stories = training_data.get_stories()
    nlu_data = training_data.get_nlu_data()
    domain = training_data.get_domain()
    flows = training_data.get_flows()
    count_conditional_responses = domain.count_conditional_response_variations()
    (
        count_total_mappings,
        count_custom_mappings,
        count_conditional_mappings,
    ) = domain.count_slot_mapping_statistics()

    training_id = uuid.uuid4().hex

    tracking_data = {
        "language": config.get("language"),
        "training_id": training_id,
        "type": model_type,
        "pipeline": config.get("pipeline"),
        "policies": config.get("policies"),
        "train_schema": config.get("train_schema"),
        "predict_schema": config.get("predict_schema"),
        "num_intent_examples": len(nlu_data.intent_examples),
        "num_entity_examples": len(nlu_data.entity_examples),
        "num_actions": len(domain.action_names_or_texts),
        # Old nomenclature from when 'responses' were still called
        # 'templates' in the domain
        "num_templates": len(domain.responses),
        "num_conditional_response_variations": count_conditional_responses,
        "num_slot_mappings": count_total_mappings,
        "num_custom_slot_mappings": count_custom_mappings,
        "num_conditional_slot_mappings": count_conditional_mappings,
        "num_slots": len(domain.slots),
        "num_forms": len(domain.forms),
        "num_intents": len(domain.intents),
        "num_entities": len(domain.entities),
        "num_story_steps": len(stories.story_steps),
        "num_lookup_tables": len(nlu_data.lookup_tables),
        "num_synonyms": len(nlu_data.entity_synonyms),
        "num_regexes": len(nlu_data.regex_features),
        "is_finetuning": is_finetuning,
        "recipe": config.get("recipe"),
    }

    flow_statistics = _collect_flow_statistics(flows.underlying_flows)
    tracking_data.update(flow_statistics)
    command_generator_settings = _get_llm_command_generator_config(config)
    tracking_data.update(command_generator_settings)

    # Make sure to update the example in docs/docs/telemetry/telemetry.mdx
    # if you change / add any properties
    _track(
        TRAINING_STARTED_EVENT,
        tracking_data,
    )
    start = datetime.now()
    yield
    runtime = datetime.now() - start

    _track(
        TRAINING_COMPLETED_EVENT,
        {
            "training_id": training_id,
            "type": model_type,
            "runtime": int(runtime.total_seconds()),
        },
    )


def _collect_flow_statistics(flows: List[Flow]) -> Dict[str, Any]:
    """Collects some statistics about the flows, such as number of specific steps."""
    data = {
        NUM_FLOWS: len(flows),
        NUM_FLOWS_WITH_NLU_TRIGGER: 0,
        NUM_FLOWS_WITH_FLOW_GUARDS: 0,
        NUM_FLOWS_ALWAYS_INCLUDED_IN_PROMPT: 0,
        NUM_FLOWS_WITH_NOT_STARTABLE_FLOW_GUARDS: 0,
        NUM_COLLECT_STEPS: 0,
        NUM_COLLECT_STEPS_WITH_SEPARATE_UTTER: 0,
        NUM_COLLECT_STEPS_WITH_REJECTIONS: 0,
        NUM_COLLECT_STEPS_WITH_NOT_RESET_AFTER_FLOW_ENDS: 0,
        NUM_SET_SLOT_STEPS: 0,
        MAX_DEPTH_OF_IF_CONSTRUCT: 0,
        NUM_LINK_STEPS: 0,
        NUM_CALL_STEPS: 0,
        NUM_SHARED_SLOTS_BETWEEN_FLOWS: 0,
    }

    slots_used_in_different_flows = defaultdict(set)

    for flow in flows:
        if flow.guard_condition:
            data[NUM_FLOWS_WITH_FLOW_GUARDS] += 1
            if flow.guard_condition.lower() == "false":
                data[NUM_FLOWS_WITH_NOT_STARTABLE_FLOW_GUARDS] += 1

        if flow.always_include_in_prompt:
            data[NUM_FLOWS_ALWAYS_INCLUDED_IN_PROMPT] += 1

        if flow.nlu_triggers:
            data[NUM_FLOWS_WITH_NLU_TRIGGER] += 1

        for step in flow.steps_with_calls_resolved:
            if isinstance(step, CollectInformationFlowStep):
                slots_used_in_different_flows[step.collect].add(flow.id)
                data[NUM_COLLECT_STEPS] += 1
                if len(step.rejections) > 0:
                    data[NUM_COLLECT_STEPS_WITH_REJECTIONS] += 1
                if not step.reset_after_flow_ends:
                    data[NUM_COLLECT_STEPS_WITH_NOT_RESET_AFTER_FLOW_ENDS] += 1
                if step.utter != f"{UTTER_ASK_PREFIX}{step.collect}":
                    data[NUM_COLLECT_STEPS_WITH_SEPARATE_UTTER] += 1

            if isinstance(step, SetSlotsFlowStep):
                for slot in step.slots:
                    slots_used_in_different_flows[slot["key"]].add(flow.id)
                data[NUM_SET_SLOT_STEPS] += 1

            if isinstance(step, LinkFlowStep):
                data[NUM_LINK_STEPS] += 1

            if isinstance(step, CallFlowStep):
                data[NUM_CALL_STEPS] += 1

            if step.next:
                depth = step.next.depth_in_tree()
                if depth > data[MAX_DEPTH_OF_IF_CONSTRUCT]:
                    data[MAX_DEPTH_OF_IF_CONSTRUCT] = depth

    for flows_with_slot in slots_used_in_different_flows.values():
        if len(flows_with_slot) > 1:
            data[NUM_SHARED_SLOTS_BETWEEN_FLOWS] += 1

    return data


def _get_llm_command_generator_config(config: Dict[str, Any]) -> Optional[Dict]:
    """Returns the configuration for the LLMCommandGenerator.

    Includes the model name, whether a custom prompt is used, whether flow
    retrieval is enabled, and flow retrieval embedding model.
    """
    from rasa.shared.constants import (
        EMBEDDINGS_CONFIG_KEY,
        MODEL_CONFIG_KEY,
        MODEL_NAME_CONFIG_KEY,
    )
    from rasa.dialogue_understanding.generator import (
        LLMCommandGenerator,
        SingleStepLLMCommandGenerator,
        MultiStepLLMCommandGenerator,
    )
    from rasa.dialogue_understanding.generator.multi_step.multi_step_llm_command_generator import (  # noqa: E501
        HANDLE_FLOWS_KEY,
        FILL_SLOTS_KEY,
    )
    from rasa.dialogue_understanding.generator.constants import (
        LLM_CONFIG_KEY,
        DEFAULT_LLM_CONFIG,
        FLOW_RETRIEVAL_KEY,
    )
    from rasa.dialogue_understanding.generator.flow_retrieval import (
        DEFAULT_EMBEDDINGS_CONFIG,
    )

    def find_command_generator_component(pipeline: List) -> Optional[Dict]:
        """Finds the LLMCommandGenerator component in the pipeline."""
        for component in pipeline:
            if component["name"] in [
                LLMCommandGenerator.__name__,
                SingleStepLLMCommandGenerator.__name__,
                MultiStepLLMCommandGenerator.__name__,
            ]:
                return component
        return None

    def extract_settings(component: Dict) -> Dict:
        """Extracts the settings from the command generator component."""
        llm_config = component.get(LLM_CONFIG_KEY, {})
        llm_model_name = (
            llm_config.get(MODEL_CONFIG_KEY)
            or llm_config.get(MODEL_NAME_CONFIG_KEY)
            or DEFAULT_LLM_CONFIG[MODEL_CONFIG_KEY]
        )
        flow_retrieval_config = component.get(FLOW_RETRIEVAL_KEY, {})
        flow_retrieval_enabled = flow_retrieval_config.get("active", True)
        flow_retrieval_embeddings_config = flow_retrieval_config.get(
            EMBEDDINGS_CONFIG_KEY, DEFAULT_EMBEDDINGS_CONFIG
        )
        flow_retrieval_embedding_model_name = (
            (
                flow_retrieval_embeddings_config.get(MODEL_NAME_CONFIG_KEY)
                or flow_retrieval_embeddings_config.get(MODEL_CONFIG_KEY)
            )
            if flow_retrieval_enabled
            else None
        )
        return {
            LLM_COMMAND_GENERATOR_MODEL_NAME: llm_model_name,
            LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED: PROMPT_CONFIG_KEY in component
            or PROMPT_TEMPLATE_CONFIG_KEY in component,
            MULTI_STEP_LLM_COMMAND_GENERATOR_HANDLE_FLOWS_PROMPT_USED: HANDLE_FLOWS_KEY
            in component.get("prompt_templates", {}),
            MULTI_STEP_LLM_COMMAND_GENERATOR_FILL_SLOTS_PROMPT_USED: FILL_SLOTS_KEY
            in component.get("prompt_templates", {}),
            FLOW_RETRIEVAL_ENABLED: flow_retrieval_enabled,
            FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME: flow_retrieval_embedding_model_name,
        }

    command_generator_config = {
        LLM_COMMAND_GENERATOR_MODEL_NAME: None,
        LLM_COMMAND_GENERATOR_CUSTOM_PROMPT_USED: None,
        MULTI_STEP_LLM_COMMAND_GENERATOR_HANDLE_FLOWS_PROMPT_USED: None,
        MULTI_STEP_LLM_COMMAND_GENERATOR_FILL_SLOTS_PROMPT_USED: None,
        FLOW_RETRIEVAL_ENABLED: None,
        FLOW_RETRIEVAL_EMBEDDING_MODEL_NAME: None,
    }

    pipeline = config.get("pipeline", [])
    if not isinstance(pipeline, list):
        return command_generator_config

    command_generator_component = find_command_generator_component(pipeline)
    if command_generator_component is not None:
        extracted_settings = extract_settings(command_generator_component)
        command_generator_config.update(extracted_settings)

    return command_generator_config


@ensure_telemetry_enabled
def track_telemetry_disabled() -> None:
    """Track when a user disables telemetry."""
    _track(TELEMETRY_DISABLED_EVENT)


@ensure_telemetry_enabled
def track_data_split(fraction: float, data_type: Text) -> None:
    """Track when a user splits data.

    Args:
        fraction: How much data goes into train and how much goes into test
        data_type: Is this core, nlu or nlg data
    """
    _track(TELEMETRY_DATA_SPLIT_EVENT, {"fraction": fraction, "type": data_type})


@ensure_telemetry_enabled
def track_validate_files(validation_success: bool) -> None:
    """Track when a user validates data files.

    Args:
        validation_success: Whether the validation was successful
    """
    _track(TELEMETRY_DATA_VALIDATED_EVENT, {"validation_success": validation_success})


@ensure_telemetry_enabled
def track_data_convert(output_format: Text, data_type: Text) -> None:
    """Track when a user converts data.

    Args:
        output_format: Target format for the converter
        data_type: Is this core, nlu or nlg data
    """
    _track(
        TELEMETRY_DATA_CONVERTED_EVENT,
        {"output_format": output_format, "type": data_type},
    )


@ensure_telemetry_enabled
def track_tracker_export(
    number_of_exported_events: int,
    tracker_store: "TrackerStore",
    event_broker: "EventBroker",
) -> None:
    """Track when a user exports trackers.

    Args:
        number_of_exported_events: Number of events that got exported
        tracker_store: Store used to retrieve the events from
        event_broker: Broker the events are getting published towards
    """
    _track(
        TELEMETRY_TRACKER_EXPORTED_EVENT,
        {
            "number_of_exported_events": number_of_exported_events,
            "tracker_store": type(tracker_store).__name__,
            "event_broker": type(event_broker).__name__,
        },
    )


@ensure_telemetry_enabled
def track_interactive_learning_start(
    skip_visualization: bool, save_in_e2e: bool
) -> None:
    """Track when a user starts an interactive learning session.

    Args:
        skip_visualization: Is visualization skipped in this session
        save_in_e2e: Is e2e used in this session
    """
    _track(
        TELEMETRY_INTERACTIVE_LEARNING_STARTED_EVENT,
        {"skip_visualization": skip_visualization, "save_in_e2e": save_in_e2e},
    )


@ensure_telemetry_enabled
def track_server_start(
    input_channels: List["InputChannel"],
    endpoints: Optional["AvailableEndpoints"],
    model_directory: Optional[Text],
    number_of_workers: int,
    is_api_enabled: bool,
) -> None:
    """Tracks when a user starts a rasa server.

    Args:
        input_channels: Used input channels
        endpoints: Endpoint configuration for the server
        model_directory: directory of the running model
        number_of_workers: number of used Sanic workers
        is_api_enabled: whether the rasa API server is enabled
    """
    from rasa.core.utils import AvailableEndpoints

    def project_fingerprint_from_model(
        _model_directory: Optional[Text],
    ) -> Optional[Text]:
        """Gets project fingerprint from an app's loaded model."""
        if not model_directory:
            return None

        try:
            model_archive = model.get_local_model(_model_directory)
            metadata = LocalModelStorage.metadata_from_archive(model_archive)

            return metadata.project_fingerprint
        except Exception:
            return None

    if not endpoints:
        endpoints = AvailableEndpoints()

    _track(
        TELEMETRY_SERVER_STARTED_EVENT,
        {
            "input_channels": [i.name() for i in input_channels],
            "api_enabled": is_api_enabled,
            "number_of_workers": number_of_workers,
            "endpoints_nlg": endpoints.nlg.type if endpoints.nlg else None,
            "endpoints_nlu": endpoints.nlu.type if endpoints.nlu else None,
            "endpoints_action_server": endpoints.action.type
            if endpoints.action
            else None,
            "endpoints_model_server": endpoints.model.type if endpoints.model else None,
            "endpoints_tracker_store": endpoints.tracker_store.type
            if endpoints.tracker_store
            else None,
            "endpoints_lock_store": endpoints.lock_store.type
            if endpoints.lock_store
            else None,
            "endpoints_event_broker": endpoints.event_broker.type
            if endpoints.event_broker
            else None,
            "project": project_fingerprint_from_model(model_directory),
        },
    )


@ensure_telemetry_enabled
def track_project_init(path: Text) -> None:
    """Track when a user creates a project using rasa init.

    Args:
        path: Location of the project
    """
    _track(
        TELEMETRY_PROJECT_CREATED_EVENT, {"init_directory": _hash_directory_path(path)}
    )


@ensure_telemetry_enabled
def track_shell_started(model_type: Text) -> None:
    """Track when a user starts a bot using rasa shell.

    Args:
        model_type: Type of the model, core / nlu or rasa.
    """
    _track(TELEMETRY_SHELL_STARTED_EVENT, {"type": model_type})


@ensure_telemetry_enabled
def track_visualization() -> None:
    """Track when a user runs the visualization."""
    _track(TELEMETRY_VISUALIZATION_STARTED_EVENT)


@ensure_telemetry_enabled
def track_core_model_test(num_story_steps: int, e2e: bool, agent: "Agent") -> None:
    """Track when a user tests a core model.

    Args:
        num_story_steps: Number of test stories used for the comparison
        e2e: indicator if tests running in end to end mode
        agent: Agent of the model getting tested
    """
    if agent.processor is None:
        project_fingerprint = ""
    else:
        project_fingerprint = agent.processor.model_metadata.project_fingerprint

    _track(
        TELEMETRY_TEST_CORE_EVENT,
        {
            "project": project_fingerprint,
            "end_to_end": e2e,
            "num_story_steps": num_story_steps,
        },
    )


@ensure_telemetry_enabled
def track_nlu_model_test(test_data: "TrainingData") -> None:
    """Track when a user tests an nlu model.

    Args:
        test_data: Data used for testing
    """
    _track(
        TELEMETRY_TEST_NLU_EVENT,
        {
            "num_intent_examples": len(test_data.intent_examples),
            "num_entity_examples": len(test_data.entity_examples),
            "num_lookup_tables": len(test_data.lookup_tables),
            "num_synonyms": len(test_data.entity_synonyms),
            "num_regexes": len(test_data.regex_features),
        },
    )


@ensure_telemetry_enabled
def track_markers_extraction_initiated(
    strategy: Text, only_extract: bool, seed: bool, count: Optional[int]
) -> None:
    """Track when a user tries to extract success markers.

    Args:
        strategy: The strategy the user is using for tracker selection
        only_extract: Indicates if the user is only extracting markers or also
                      producing stats
        seed: Indicates if the user used a seed for this attempt
        count: (Optional) The number of trackers the user is trying to select.
    """
    _track(
        TELEMETRY_MARKERS_EXTRACTION_INITIATED_EVENT,
        {
            "strategy": strategy,
            "only_extract": only_extract,
            "seed": seed,
            "count": count,
        },
    )


@ensure_telemetry_enabled
def track_markers_extracted(trackers_count: int) -> None:
    """Track when markers have been extracted by a user.

    Args:
        trackers_count: The actual number of trackers processed
    """
    _track(TELEMETRY_MARKERS_EXTRACTED_EVENT, {"trackers_count": trackers_count})


@ensure_telemetry_enabled
def track_markers_stats_computed(trackers_count: int) -> None:
    """Track when stats over markers have been computed by a user.

    Args:
        trackers_count: The actual number of trackers processed
    """
    _track(TELEMETRY_MARKERS_STATS_COMPUTED_EVENT, {"trackers_count": trackers_count})


@ensure_telemetry_enabled
def track_markers_parsed_count(
    marker_count: int, max_depth: int, branching_factor: int
) -> None:
    """Track when markers have been successfully parsed from config.

    Args:
        marker_count: The number of markers found in the config
        max_depth: The maximum depth of any marker in the config
        branching_factor: The maximum number of children of any marker in the config.
    """
    _track(
        TELEMETRY_MARKERS_PARSED_COUNT,
        {
            "marker_count": marker_count,
            "max_depth": max_depth,
            "branching_factor": branching_factor,
        },
    )


def extract_assertion_type_counts(
    input_test_cases: List["TestCase"],
) -> typing.Tuple[bool, Dict[str, Any]]:
    """Extracts the total count of different assertion types from the test cases."""
    from rasa.e2e_test.assertions import AssertionType

    uses_assertions = False

    flow_started_count = 0
    flow_completed_count = 0
    flow_cancelled_count = 0
    pattern_clarification_contains_count = 0
    action_executed_count = 0
    slot_was_set_count = 0
    slot_was_not_set_count = 0
    bot_uttered_count = 0
    generative_response_is_relevant_count = 0
    generative_response_is_grounded_count = 0

    for test_case in input_test_cases:
        for step in test_case.steps:
            assertions = step.assertions if step.assertions else []
            for assertion in assertions:
                if assertion.type == AssertionType.ACTION_EXECUTED.value:
                    action_executed_count += 1
                elif assertion.type == AssertionType.SLOT_WAS_SET.value:
                    slot_was_set_count += 1
                elif assertion.type == AssertionType.SLOT_WAS_NOT_SET.value:
                    slot_was_not_set_count += 1
                elif assertion.type == AssertionType.BOT_UTTERED.value:
                    bot_uttered_count += 1
                elif (
                    assertion.type
                    == AssertionType.GENERATIVE_RESPONSE_IS_RELEVANT.value
                ):
                    generative_response_is_relevant_count += 1
                elif (
                    assertion.type
                    == AssertionType.GENERATIVE_RESPONSE_IS_GROUNDED.value
                ):
                    generative_response_is_grounded_count += 1
                elif assertion.type == AssertionType.FLOW_STARTED.value:
                    flow_started_count += 1
                elif assertion.type == AssertionType.FLOW_COMPLETED.value:
                    flow_completed_count += 1
                elif assertion.type == AssertionType.FLOW_CANCELLED.value:
                    flow_cancelled_count += 1
                elif (
                    assertion.type == AssertionType.PATTERN_CLARIFICATION_CONTAINS.value
                ):
                    pattern_clarification_contains_count += 1

                uses_assertions = True

    result = {
        "flow_started_count": flow_started_count,
        "flow_completed_count": flow_completed_count,
        "flow_cancelled_count": flow_cancelled_count,
        "pattern_clarification_contains_count": pattern_clarification_contains_count,
        "action_executed_count": action_executed_count,
        "slot_was_set_count": slot_was_set_count,
        "slot_was_not_set_count": slot_was_not_set_count,
        "bot_uttered_count": bot_uttered_count,
        "generative_response_is_relevant_count": generative_response_is_relevant_count,
        "generative_response_is_grounded_count": generative_response_is_grounded_count,
    }

    return uses_assertions, result


@ensure_telemetry_enabled
def track_e2e_test_run(
    input_test_cases: List["TestCase"],
    input_fixtures: List["Fixture"],
    input_metadata: List["Metadata"],
) -> None:
    """Track an end-to-end test run."""
    properties = {
        "number_of_test_cases": len(input_test_cases),
        "number_of_fixtures": len(input_fixtures),
        "uses_fixtures": len(input_fixtures) > 0,
        "uses_metadata": len(input_metadata) > 0,
        "number_of_metadata": len(input_metadata),
    }

    uses_assertions, assertion_type_counts = extract_assertion_type_counts(
        input_test_cases
    )

    properties.update({"uses_assertions": uses_assertions})

    if uses_assertions:
        properties.update(assertion_type_counts)

    _track(
        TELEMETRY_E2E_TEST_RUN_STARTED_EVENT,
        properties,
    )


@ensure_telemetry_enabled
def track_response_rephrase(
    rephrase_all: bool,
    custom_prompt_template: Optional[str],
    llm_type: Optional[str],
    llm_model: Optional[str],
) -> None:
    """Track when a user rephrases a response."""
    _track(
        TELEMETRY_RESPONSE_REPHRASED_EVENT,
        {
            "rephrase_all": rephrase_all,
            "custom_prompt_template": custom_prompt_template,
            "llm_type": llm_type,
            "llm_model": llm_model,
        },
    )


@ensure_telemetry_enabled
def track_intentless_policy_train() -> None:
    """Track when a user trains a policy."""
    _track(TELEMETRY_INTENTLESS_POLICY_TRAINING_STARTED_EVENT)


@ensure_telemetry_enabled
def track_intentless_policy_train_completed(
    embeddings_type: Optional[str],
    embeddings_model: Optional[str],
    llm_type: Optional[str],
    llm_model: Optional[str],
) -> None:
    """Track when a user trains a policy."""
    _track(
        TELEMETRY_INTENTLESS_POLICY_TRAINING_COMPLETED_EVENT,
        {
            "embeddings_type": embeddings_type,
            "embeddings_model": embeddings_model,
            "llm_type": llm_type,
            "llm_model": llm_model,
        },
    )


@ensure_telemetry_enabled
def track_intentless_policy_predict(
    embeddings_type: Optional[str],
    embeddings_model: Optional[str],
    llm_type: Optional[str],
    llm_model: Optional[str],
    score: float,
) -> None:
    """Track when a user trains a policy."""
    _track(
        TELEMETRY_INTENTLESS_POLICY_PREDICT_EVENT,
        {
            "embeddings_type": embeddings_type,
            "embeddings_model": embeddings_model,
            "llm_type": llm_type,
            "llm_model": llm_model,
            "score": score,
        },
    )


@ensure_telemetry_enabled
def identify_endpoint_config_traits(
    endpoints_file: Optional[Text],
    context: Optional[Dict[Text, Any]] = None,
) -> None:
    """Collect traits if enabled.

    Otherwise, sets traits to None.
    """
    traits: Dict[str, Any] = {}

    traits = append_tracing_trait(traits, endpoints_file)
    traits = append_metrics_trait(traits, endpoints_file)
    traits = append_anonymization_trait(traits, endpoints_file)

    _identify(traits, context)


def append_tracing_trait(
    traits: Dict[str, Any], endpoints_file: Optional[str]
) -> Dict[str, Any]:
    """Append the tracing trait to the traits dictionary."""
    import rasa.utils.endpoints
    from rasa.tracing.constants import ENDPOINTS_TRACING_KEY

    tracing_config = rasa.utils.endpoints.read_endpoint_config(
        endpoints_file, ENDPOINTS_TRACING_KEY
    )
    traits[TRACING_BACKEND] = (
        tracing_config.type if tracing_config is not None else None
    )

    return traits


def append_metrics_trait(
    traits: Dict[str, Any], endpoints_file: Optional[str]
) -> Dict[str, Any]:
    """Append the metrics trait to the traits dictionary."""
    import rasa.utils.endpoints
    from rasa.tracing.constants import ENDPOINTS_METRICS_KEY

    metrics_config = rasa.utils.endpoints.read_endpoint_config(
        endpoints_file, ENDPOINTS_METRICS_KEY
    )
    traits[METRICS_BACKEND] = (
        metrics_config.type if metrics_config is not None else None
    )

    return traits


def append_anonymization_trait(
    traits: Dict[str, Any], endpoints_file: Optional[str]
) -> Dict[str, Any]:
    """Append the anonymization trait to the traits dictionary."""
    from rasa.anonymization.anonymisation_rule_yaml_reader import (
        KEY_ANONYMIZATION_RULES,
    )

    anonymization_config = rasa.anonymization.utils.read_endpoint_config(
        endpoints_file, KEY_ANONYMIZATION_RULES
    )

    traits[KEY_ANONYMIZATION_RULES] = (
        rasa.anonymization.utils.extract_anonymization_traits(
            anonymization_config, KEY_ANONYMIZATION_RULES
        )
    )

    return traits


@ensure_telemetry_enabled
def track_enterprise_search_policy_train_started() -> None:
    """Track when a user starts training Enterprise Search policy."""
    _track(TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_STARTED_EVENT)


@ensure_telemetry_enabled
def track_enterprise_search_policy_train_completed(
    vector_store_type: Optional[str],
    embeddings_type: Optional[str],
    embeddings_model: Optional[str],
    llm_type: Optional[str],
    llm_model: Optional[str],
    citation_enabled: Optional[bool],
) -> None:
    """Track when a user completes training Enterprise Search policy."""
    _track(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_TRAINING_COMPLETED_EVENT,
        {
            "vector_store_type": vector_store_type,
            "embeddings_type": embeddings_type,
            "embeddings_model": embeddings_model,
            "llm_type": llm_type,
            "llm_model": llm_model,
            "citation_enabled": citation_enabled,
        },
    )


@ensure_telemetry_enabled
def track_enterprise_search_policy_predict(
    vector_store_type: Optional[str],
    embeddings_type: Optional[str],
    embeddings_model: Optional[str],
    llm_type: Optional[str],
    llm_model: Optional[str],
    citation_enabled: Optional[bool],
) -> None:
    """Track when a user predicts the next action using Enterprise Search policy."""
    _track(
        TELEMETRY_ENTERPRISE_SEARCH_POLICY_PREDICT_EVENT,
        {
            "vector_store_type": vector_store_type,
            "embeddings_type": embeddings_type,
            "embeddings_model": embeddings_model,
            "llm_type": llm_type,
            "llm_model": llm_model,
            "citation_enabled": citation_enabled,
        },
    )


@ensure_telemetry_enabled
def track_conversation_count_hard_limit(
    conversation_count: int, tracked_month: datetime
) -> None:
    """Track when the number of conversations reaches the hard limit."""
    _track(
        TELEMETRY_CONVERSATION_HARD_LIMIT_REACHED,
        {
            "conversation_count": conversation_count,
            "year": tracked_month.year,
            "month": tracked_month.month,
        },
    )


@ensure_telemetry_enabled
def track_conversation_count_soft_limit(
    conversation_count: int, tracked_month: datetime
) -> None:
    """Track when the number of conversations reaches the soft limit."""
    _track(
        TELEMETRY_CONVERSATION_SOFT_LIMIT_REACHED,
        {
            "conversation_count": conversation_count,
            "year": tracked_month.year,
            "month": tracked_month.month,
        },
    )


@ensure_telemetry_enabled
def track_conversation_count(conversation_count: int, tracked_month: datetime) -> None:
    """Track the number of conversations."""
    _track(
        TELEMETRY_CONVERSATION_COUNT,
        {
            "conversation_count": conversation_count,
            "year": tracked_month.year,
            "month": tracked_month.month,
        },
    )


@ensure_telemetry_enabled
def track_e2e_test_conversion_completed(file_type: str, test_case_count: int) -> None:
    """Track the used input file type for E2E test conversion."""
    _track(
        TELEMETRY_E2E_TEST_CONVERSION_EVENT,
        {
            E2E_TEST_CONVERSION_FILE_TYPE: file_type,
            E2E_TEST_CONVERSION_TEST_CASE_COUNT: test_case_count,
        },
    )
