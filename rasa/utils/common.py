import logging
import os
import shutil
import warnings
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Text, Type, Collection

import rasa.core.utils
import rasa.utils.io
from rasa.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_LEVEL_LIBRARIES,
    ENV_LOG_LEVEL,
    ENV_LOG_LEVEL_LIBRARIES,
    GLOBAL_USER_CONFIG_PATH,
    NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
)
import rasa.shared.utils.io

logger = logging.getLogger(__name__)


class TempDirectoryPath(str):
    """Represents a path to an temporary directory. When used as a context
    manager, it erases the contents of the directory on exit.

    """

    def __enter__(self) -> "TempDirectoryPath":
        return self

    def __exit__(
        self,
        _exc: Optional[Type[BaseException]],
        _value: Optional[Exception],
        _tb: Optional[TracebackType],
    ) -> bool:
        if os.path.exists(self):
            shutil.rmtree(self)


def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())


def read_global_config() -> Dict[Text, Any]:
    """Read global Rasa configuration."""
    # noinspection PyBroadException
    try:
        return rasa.utils.io.read_config_file(GLOBAL_USER_CONFIG_PATH)
    except Exception:
        # if things go south we pretend there is no config
        return {}


def set_log_level(log_level: Optional[int] = None):
    """Set log level of Rasa and Tensorflow either to the provided log level or
    to the log level specified in the environment variable 'LOG_LEVEL'. If none is set
    a default log level will be used."""

    if not log_level:
        log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        log_level = logging.getLevelName(log_level)

    logging.getLogger("rasa").setLevel(log_level)

    update_tensorflow_log_level()
    update_asyncio_log_level()
    update_apscheduler_log_level()
    update_socketio_log_level()

    os.environ[ENV_LOG_LEVEL] = logging.getLevelName(log_level)


def update_apscheduler_log_level() -> None:
    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    apscheduler_loggers = [
        "apscheduler",
        "apscheduler.scheduler",
        "apscheduler.executors",
        "apscheduler.executors.default",
    ]

    for logger_name in apscheduler_loggers:
        logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger(logger_name).propagate = False


def update_socketio_log_level() -> None:
    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    socketio_loggers = ["websockets.protocol", "engineio.server", "socketio.server"]

    for logger_name in socketio_loggers:
        logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger(logger_name).propagate = False


def update_tensorflow_log_level() -> None:
    """Set the log level of Tensorflow to the log level specified in the environment
    variable 'LOG_LEVEL_LIBRARIES'."""

    # Disables libvinfer, tensorRT, cuda, AVX2 and FMA warnings (CPU support). This variable needs to be set before the
    # first import since some warnings are raised on the first import.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    import tensorflow as tf

    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    if log_level == "DEBUG":
        tf_log_level = tf.compat.v1.logging.DEBUG
    elif log_level == "INFO":
        tf_log_level = tf.compat.v1.logging.INFO
    elif log_level == "WARNING":
        tf_log_level = tf.compat.v1.logging.WARN
    else:
        tf_log_level = tf.compat.v1.logging.ERROR

    tf.compat.v1.logging.set_verbosity(tf_log_level)
    logging.getLogger("tensorflow").propagate = False


def update_sanic_log_level(log_file: Optional[Text] = None):
    """Set the log level of sanic loggers to the log level specified in the environment
    variable 'LOG_LEVEL_LIBRARIES'."""
    from sanic.log import logger, error_logger, access_logger

    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    logger.setLevel(log_level)
    error_logger.setLevel(log_level)
    access_logger.setLevel(log_level)

    logger.propagate = False
    error_logger.propagate = False
    access_logger.propagate = False

    if log_file is not None:
        formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        error_logger.addHandler(file_handler)
        access_logger.addHandler(file_handler)


def update_asyncio_log_level() -> None:
    """Set the log level of asyncio to the log level specified in the environment
    variable 'LOG_LEVEL_LIBRARIES'."""
    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)
    logging.getLogger("asyncio").setLevel(log_level)


def set_log_and_warnings_filters() -> None:
    """
    Set log filters on the root logger, and duplicate filters for warnings.

    Filters only propagate on handlers, not loggers.
    """
    for handler in logging.getLogger().handlers:
        handler.addFilter(RepeatedLogFilter())

    warnings.filterwarnings("once", category=UserWarning)


def obtain_verbosity() -> int:
    """Returns a verbosity level according to the set log level."""
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    verbosity = 0
    if log_level == "DEBUG":
        verbosity = 2
    if log_level == "INFO":
        verbosity = 1

    return verbosity


def is_logging_disabled() -> bool:
    """Returns true, if log level is set to WARNING or ERROR, false otherwise."""
    log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)

    return log_level == "ERROR" or log_level == "WARNING"


def sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]:
    """Sorts a list of dictionaries by their first key."""
    return sorted(dicts, key=lambda d: list(d.keys())[0])


def transform_collection_to_sentence(collection: Collection[Text]) -> Text:
    """Transforms e.g. a list like ['A', 'B', 'C'] into a sentence 'A, B and C'."""
    x = list(collection)
    if len(x) >= 2:
        return ", ".join(map(str, x[:-1])) + " and " + x[-1]
    return "".join(collection)


def minimal_kwargs(
    kwargs: Dict[Text, Any], func: Callable, excluded_keys: Optional[List] = None
) -> Dict[Text, Any]:
    """Returns only the kwargs which are required by a function. Keys, contained in
    the exception list, are not included.

    Args:
        kwargs: All available kwargs.
        func: The function which should be called.
        excluded_keys: Keys to exclude from the result.

    Returns:
        Subset of kwargs which are accepted by `func`.

    """

    excluded_keys = excluded_keys or []

    possible_arguments = arguments_of(func)

    return {
        k: v
        for k, v in kwargs.items()
        if k in possible_arguments and k not in excluded_keys
    }


def write_global_config_value(name: Text, value: Any) -> None:
    """Read global Rasa configuration."""

    try:
        os.makedirs(os.path.dirname(GLOBAL_USER_CONFIG_PATH), exist_ok=True)

        c = read_global_config()
        c[name] = value
        rasa.core.utils.dump_obj_as_yaml_to_file(GLOBAL_USER_CONFIG_PATH, c)
    except Exception as e:
        logger.warning(f"Failed to write global config. Error: {e}. Skipping.")


def read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any:
    """Read a value from the global Rasa configuration."""

    def not_found():
        if unavailable_ok:
            return None
        else:
            raise ValueError(f"Configuration '{name}' key not found.")

    if not os.path.exists(GLOBAL_USER_CONFIG_PATH):
        return not_found()

    c = read_global_config()

    if name in c:
        return c[name]
    else:
        return not_found()


def mark_as_experimental_feature(feature_name: Text) -> None:
    """Warns users that they are using an experimental feature."""

    logger.warning(
        f"The {feature_name} is currently experimental and might change or be "
        "removed in the future 🔬 Please share your feedback on it in the "
        "forum (https://forum.rasa.com) to help us make this feature "
        "ready for production."
    )


def update_existing_keys(
    original: Dict[Any, Any], updates: Dict[Any, Any]
) -> Dict[Any, Any]:
    """Iterate through all the updates and update a value in the original dictionary.

    If the updates contain a key that is not present in the original dict, it will
    be ignored."""

    updated = original.copy()
    for k, v in updates.items():
        if k in updated:
            updated[k] = v
    return updated


def raise_deprecation_warning(
    message: Text,
    warn_until_version: Text = NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
    docs: Optional[Text] = None,
    **kwargs: Any,
) -> None:
    """
    Thin wrapper around `raise_warning()` to raise a deprecation warning. It requires
    a version until which we'll warn, and after which the support for the feature will
    be removed.
    """
    if warn_until_version not in message:
        message = f"{message} (will be removed in {warn_until_version})"

    # need the correct stacklevel now
    kwargs.setdefault("stacklevel", 3)
    # we're raising a `FutureWarning` instead of a `DeprecationWarning` because
    # we want these warnings to be visible in the terminal of our users
    # https://docs.python.org/3/library/warnings.html#warning-categories
    rasa.shared.utils.io.raise_warning(message, FutureWarning, docs, **kwargs)


class RepeatedLogFilter(logging.Filter):
    """Filter repeated log records."""

    last_log = None

    def filter(self, record):
        current_log = (
            record.levelno,
            record.pathname,
            record.lineno,
            record.msg,
            record.args,
        )
        if current_log != self.last_log:
            self.last_log = current_log
            return True
        return False
