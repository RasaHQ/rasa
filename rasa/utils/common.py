import os
from typing import Any, Callable, Dict, List, Text

import rasa.core.utils
import rasa.utils.io
from rasa.constants import GLOBAL_USER_CONFIG_PATH, DEFAULT_LOG_LEVEL


def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())


def read_global_config() -> Dict[Text, Any]:
    """Read global Rasa configuration."""
    # noinspection PyBroadException
    try:
        return rasa.utils.io.read_yaml_file(GLOBAL_USER_CONFIG_PATH)
    except Exception:
        # if things go south we pretend there is no config
        return {}


def write_global_config_value(name: Text, value: Any) -> None:
    """Read global Rasa configuration."""

    os.makedirs(os.path.dirname(GLOBAL_USER_CONFIG_PATH), exist_ok=True)

    c = read_global_config()
    c[name] = value
    rasa.core.utils.dump_obj_as_yaml_to_file(GLOBAL_USER_CONFIG_PATH, c)


def read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any:
    """Read a value from the global Rasa configuration."""

    def not_found():
        if unavailable_ok:
            return None
        else:
            raise ValueError("Configuration '{}' key not found.".format(name))

    if not os.path.exists(GLOBAL_USER_CONFIG_PATH):
        return not_found()

    c = read_global_config()

    if name in c:
        return c[name]
    else:
        return not_found()


def set_tensorflow_log_level():
    import tensorflow as tf

    log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    tf_log_level = tf.logging.INFO
    if log_level == "DEBUG":
        tf_log_level = tf.logging.DEBUG
    if log_level == "WARNING":
        tf_log_level = tf.logging.WARN
    if log_level == "ERROR":
        tf_log_level = tf.logging.ERROR

    tf.logging.set_verbosity(tf_log_level)


def set_sanic_log_level():
    from sanic.log import logger as sanic_logger

    log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    sanic_logger.setLevel(log_level)


def obtain_verbosity():
    log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    verbosity = 0
    if log_level == "DEBUG":
        verbosity = 2
    if log_level == "INFO":
        verbosity = 1

    return verbosity


def disable_logging():
    log_level = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL)

    return not (log_level == "DEBUG" or log_level == "INFO")
