import os
from typing import Any, Callable, Dict, List, Text

import rasa.core.utils
import rasa.utils.io
from rasa.constants import GLOBAL_USER_CONFIG_PATH


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
