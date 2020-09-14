import asyncio
import async_generator
from datetime import datetime
import functools
from functools import wraps
import hashlib
import json
import logging
import multiprocessing
import os
from pathlib import Path
import platform
from subprocess import CalledProcessError, STDOUT, check_output  # skipcq:BAN-B404
import sys
import textwrap
from typing import Any, Callable, Dict, Optional, Text
import uuid

import aiohttp
from terminaltables import SingleTable

import rasa
from rasa.shared.importers.importer import TrainingDataImporter
import rasa.shared.utils.io
from rasa.constants import (
    CONFIG_FILE_TELEMETRY_KEY,
    CONFIG_TELEMETRY_DATE,
    CONFIG_TELEMETRY_ENABLED,
    CONFIG_TELEMETRY_ID,
    DOCS_URL_TELEMETRY,
)
from rasa.utils import common as rasa_utils
import rasa.utils.io

logger = logging.getLogger(__name__)

SEGMENT_ENDPOINT = "https://api.segment.io/v1/track"

TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE = "RASA_TELEMETRY_ENABLED"
TELEMETRY_DEBUG_ENVIRONMENT_VARIABLE = "RASA_TELEMETRY_DEBUG"

# the environment variable can be used for local development to set a test key
# e.g. `RASA_TELEMETRY_WRITE_KEY=12354 rasa train`
TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE = "RASA_TELEMETRY_WRITE_KEY"

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
]

# If updating or creating a new event, remember to update
# https://rasa.com/docs/rasa/telemetry
TRAINING_STARTED_EVENT = "Training Started"
TRAINING_COMPLETED_EVENT = "Training Completed"
TELEMETRY_DISABLED = "Telemetry Disabled"


def print_telemetry_reporting_info() -> None:
    """Print telemetry information to std out."""
    message = textwrap.dedent(
        f"""
      Rasa reports anonymous usage telemetry to help improve Rasa Open Source
      for all its users.

      If you'd like to opt-out, you can use `rasa telemetry disable`.
      To learn more, check out {DOCS_URL_TELEMETRY}."""
    ).strip()

    table = SingleTable([[message]])
    print(table.table)


def _default_telemetry_configuration(is_enabled: bool) -> Dict[Text, Text]:
    return {
        CONFIG_TELEMETRY_ENABLED: is_enabled,
        CONFIG_TELEMETRY_ID: uuid.uuid4().hex,
        CONFIG_TELEMETRY_DATE: datetime.now(),
    }


def _write_default_telemetry_configuration(
    is_enabled: bool = TELEMETRY_ENABLED_BY_DEFAULT,
) -> None:
    if is_enabled:
        print_telemetry_reporting_info()

    new_config = _default_telemetry_configuration(is_enabled)

    rasa_utils.write_global_config_value(CONFIG_FILE_TELEMETRY_KEY, new_config)


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

        # seems like there is no config, we'll create on and enable telemetry
        _write_default_telemetry_configuration()
        return TELEMETRY_ENABLED_BY_DEFAULT


def initialize_telemetry() -> bool:
    """Read telemetry configuration from the user's Rasa config file in $HOME.

    Creates a default configuration if no configuration exists.

    Returns:
        `True`, if telemetry is enabled, `False` otherwise.
    """

    # calling this even if the environment variable is set makes sure the
    # configuration is created and there is a telemetry ID
    is_enabled_in_configuration = _is_telemetry_enabled_in_configuration()

    telemetry_environ = os.environ.get(TELEMETRY_ENABLED_ENVIRONMENT_VARIABLE)

    if telemetry_environ is None:
        return is_enabled_in_configuration
    else:
        return telemetry_environ.lower() == "true"


def ensure_telemetry_enabled(f: Callable[..., Any]) -> Callable[..., Any]:
    """Function decorator for telemetry functions that only runs the decorated
    function if telemetry is enabled."""

    is_telemetry_enabled = initialize_telemetry()

    # allows us to use the decorator for async and non async functions
    if asyncio.iscoroutinefunction(f):

        @wraps(f)
        async def decorated(*args, **kwargs):
            try:
                if is_telemetry_enabled:
                    return await f(*args, **kwargs)
            except Exception as e:  # skipcq:PYL-W0703
                logger.debug(f"Skipping telemetry reporting: {e}")
            return None

        return decorated
    else:

        @wraps(f)
        def decorated(*args, **kwargs):
            try:
                if is_telemetry_enabled:
                    return f(*args, **kwargs)
            except Exception as e:  # skipcq:PYL-W0703
                logger.debug(f"Skipping telemetry reporting: {e}")
            return None

        return decorated


def telemetry_write_key() -> Optional[Text]:
    """Read the Segment write key from the segment key text file.
    The segment key text file should by present only in wheel/sdist packaged
    versions of Rasa Open Source. This avoids running telemetry locally when
    developing on Rasa or when running CI builds.

    In local development, this should always return `None` to avoid logging telemetry.

    Returns:
        Segment write key, if the key file was present.
    """
    import pkg_resources
    from rasa import __name__ as name

    if os.environ.get(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE):
        # a write key set using the environment variable will always
        # overwrite any key provided as part of the package (`segment_key` file)
        return os.environ.get(TELEMETRY_WRITE_KEY_ENVIRONMENT_VARIABLE)

    write_key_path = pkg_resources.resource_filename(name, "segment_key")

    # noinspection PyBroadException
    try:
        with open(write_key_path) as f:
            return f.read().strip()
    except Exception:  # skipcq:PYL-W0703
        return None


def encode_base64(original: Text, encoding: Text = "utf-8") -> Text:
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
        "Authorization": "Basic {}".format(encode_base64(write_key + ":")),
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


def print_telemetry_event(payload: Dict[Text, Any]) -> None:
    """Print a telemetry events payload to the commandline.

    Args:
        payload: payload of the event
    """
    print("Telemetry Event:")
    print(json.dumps(payload, indent=2))


async def _send_event(
    distinct_id: Text,
    event_name: Text,
    properties: Dict[Text, Any],
    context: Dict[Text, Any],
) -> None:
    """Report the contents of an event to the /track Segment endpoint.
    Documentation: https://segment.com/docs/sources/server/http/

    Do not call this function from outside telemetry.py! This function does not
    check if telemetry is enabled or not.

    Args:
        distinct_id: Unique telemetry ID.
        event_name: Name of the event.
        properties: Values to report along the event.
        context: Context information about the event.
    """

    payload = segment_request_payload(distinct_id, event_name, properties, context)

    if _is_telemetry_debug_enabled():
        print_telemetry_event(payload)
        return

    write_key = telemetry_write_key()
    if not write_key:
        # If TELEMETRY_WRITE_KEY is empty or `None`, telemetry has not been
        # enabled for this build (e.g. because it is running from source)
        logger.debug("Skipping request to external service: telemetry key not set.")
        return

    headers = segment_request_header(write_key)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            SEGMENT_ENDPOINT, headers=headers, json=payload,
        ) as resp:
            # handle different failure cases
            if resp.status != 200:
                response_text = await resp.text()
                logger.debug(
                    f"Segment telemetry request returned a {resp.status} response."
                    f"Body: {response_text}"
                )
            else:
                data = await resp.json()
                if not data.get("success"):
                    logger.debug(
                        f"Segment telemetry request returned a failure."
                        f"Response: {data}"
                    )


def _project_hash() -> Text:
    """Create a hash for the project in the current working directory.

    Returns:
        project hash
    """
    try:
        remote = check_output(  # skipcq:BAN-B607,BAN-B603
            ["git", "remote", "get-url", "origin"], stderr=STDOUT
        )
        return hashlib.sha256(remote).hexdigest()
    except (CalledProcessError, OSError):
        working_dir = Path(os.getcwd()).absolute()
        return hashlib.sha256(str(working_dir).encode("utf-8")).hexdigest()


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
    """Return a new context dictionary that contains the default field values merged
    with the provided ones. The default fields contain only the OS information for now.

    Args:
        context: Context information about the event.

    Return:
        A new context.
    """
    context = context or {}

    return {**_default_context_fields(), **context}


@functools.lru_cache()
def _default_context_fields() -> Dict[Text, Any]:
    """Return a dictionary that contains the default context values.

    Return:
        A new context containing information about the runtime environment.
    """
    import tensorflow as tf

    return {
        "os": {"name": platform.system(), "version": platform.release()},
        "ci": in_continuous_integration(),
        "project": _project_hash(),
        "python": sys.version.split(" ")[0],
        "rasa_open_source": rasa.__version__,
        "gpu": len(tf.config.list_physical_devices("GPU")),
        "cpu": multiprocessing.cpu_count(),
        "docker": _is_docker(),
    }


@ensure_telemetry_enabled
async def track(
    event_name: Text,
    properties: Optional[Dict[Text, Any]] = None,
    context: Optional[Dict[Text, Any]] = None,
) -> None:
    """Tracks a telemetry event.

    It is OK to use this function from outside telemetry.py, but note that it
    is recommended to create a new track_xyz() function for complex telemetry
    events, or events that are generated from many parts of the Rasa Open Source code.

    Args:
        event_name: Name of the event.
        properties: Dictionary containing the event's properties.
        context: Dictionary containing some context for this event.
    """

    telemetry_id = get_telemetry_id()

    if not telemetry_id:
        logger.debug("Will not report telemetry events as no ID was found.")
        return

    if not properties:
        properties = {}

    properties[TELEMETRY_ID] = telemetry_id

    await _send_event(
        telemetry_id, event_name, properties, with_default_context_fields(context)
    )


def get_telemetry_id() -> Optional[Text]:
    """Return the unique telemetry identifier for this Rasa Open Source install.
    The identifier can be any string, but it should be a UUID.

    Returns:
        The identifier, if it is configured correctly.
    """

    try:
        telemetry_config = (
            rasa_utils.read_global_config_value(CONFIG_FILE_TELEMETRY_KEY) or {}
        )

        return telemetry_config.get(CONFIG_TELEMETRY_ID)
    except Exception as e:  # skipcq:PYL-W0703
        logger.debug(f"Unable to retrieve telemetry ID: {e}")
        return None


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


@async_generator.asynccontextmanager
@async_generator.async_generator
@ensure_telemetry_enabled
async def track_model_training(
    training_data: TrainingDataImporter, model_type: Text
) -> None:
    """Track a model training started.

    Args:
        training_data: Training data used for the training.
        model_type: Specifies the type of training, should be either "rasa", "core"
            or "nlu".
    """

    config = await training_data.get_config()
    stories = await training_data.get_stories()
    nlu_data = await training_data.get_nlu_data()
    domain = await training_data.get_domain()

    training_id = uuid.uuid4().hex

    asyncio.ensure_future(
        track(
            TRAINING_STARTED_EVENT,
            {
                "language": config.get("language"),
                "training_id": training_id,
                "model_type": model_type,
                "pipeline": config.get("pipeline"),
                "policies": config.get("policies"),
                "num_intent_examples": len(nlu_data.intent_examples),
                "num_entity_examples": len(nlu_data.entity_examples),
                "num_actions": len(domain.action_names),
                # Old nomenclature from when 'responses' were still called
                # 'templates' in the domain
                "num_templates": len(domain.templates),
                "num_slots": len(domain.slots),
                "num_forms": len(domain.forms),
                "num_intents": len(domain.intents),
                "num_entities": len(domain.entities),
                "num_story_steps": len(stories.story_steps),
                "num_lookup_tables": len(nlu_data.lookup_tables),
                "num_synonyms": len(nlu_data.entity_synonyms),
                "num_regexes": len(nlu_data.regex_features),
            },
        )
    )
    start = datetime.now()
    yield
    runtime = datetime.now() - start

    asyncio.ensure_future(
        track(
            TRAINING_COMPLETED_EVENT,
            {
                "training_id": training_id,
                "model_type": model_type,
                "runtime": int(runtime.total_seconds()),
            },
        )
    )


@ensure_telemetry_enabled
async def track_telemetry_disabled() -> None:
    """Track when a user disables telemetry."""

    asyncio.ensure_future(track(TELEMETRY_DISABLED))
