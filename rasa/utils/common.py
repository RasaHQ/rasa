import asyncio
import logging
import os
import shutil
import warnings
from types import TracebackType
from typing import (
    Any,
    Coroutine,
    Dict,
    List,
    Optional,
    Text,
    Type,
    TypeVar,
    Union,
)

import rasa.utils.io
from rasa.constants import DEFAULT_LOG_LEVEL_LIBRARIES, ENV_LOG_LEVEL_LIBRARIES
from rasa.shared.constants import DEFAULT_LOG_LEVEL, ENV_LOG_LEVEL
import rasa.shared.utils.io

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


def read_global_config(path: Text) -> Dict[Text, Any]:
    """Read global Rasa configuration.

    Args:
        path: Path to the configuration
    Returns:
        The global configuration
    """
    # noinspection PyBroadException
    try:
        return rasa.shared.utils.io.read_config_file(path)
    except Exception:
        # if things go south we pretend there is no config
        return {}


def set_log_level(log_level: Optional[int] = None) -> None:
    """Set log level of Rasa and Tensorflow either to the provided log level or
    to the log level specified in the environment variable 'LOG_LEVEL'. If none is set
    a default log level will be used."""

    if not log_level:
        log_level = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        log_level = logging.getLevelName(log_level)

    logging.getLogger("rasa").setLevel(log_level)

    update_tensorflow_log_level()
    update_asyncio_log_level()
    update_matplotlib_log_level()
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
    """Sets Tensorflow log level based on env variable 'LOG_LEVEL_LIBRARIES'."""
    # Disables libvinfer, tensorRT, cuda, AVX2 and FMA warnings (CPU support).
    # This variable needs to be set before the
    # first import since some warnings are raised on the first import.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)

    if not log_level:
        log_level = "ERROR"

    logging.getLogger("tensorflow").setLevel(log_level)
    logging.getLogger("tensorflow").propagate = False


def update_sanic_log_level(log_file: Optional[Text] = None) -> None:
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


def update_matplotlib_log_level() -> None:
    """Set the log level of matplotlib to the log level specified in the environment
    variable 'LOG_LEVEL_LIBRARIES'."""
    log_level = os.environ.get(ENV_LOG_LEVEL_LIBRARIES, DEFAULT_LOG_LEVEL_LIBRARIES)
    logging.getLogger("matplotlib.backends.backend_pdf").setLevel(log_level)
    logging.getLogger("matplotlib.colorbar").setLevel(log_level)
    logging.getLogger("matplotlib.font_manager").setLevel(log_level)
    logging.getLogger("matplotlib.ticker").setLevel(log_level)


def set_log_and_warnings_filters() -> None:
    """
    Set log filters on the root logger, and duplicate filters for warnings.

    Filters only propagate on handlers, not loggers.
    """
    for handler in logging.getLogger().handlers:
        handler.addFilter(RepeatedLogFilter())

    warnings.filterwarnings("once", category=UserWarning)


def sort_list_of_dicts_by_first_key(dicts: List[Dict]) -> List[Dict]:
    """Sorts a list of dictionaries by their first key."""
    return sorted(dicts, key=lambda d: list(d.keys())[0])


def write_global_config_value(name: Text, value: Any) -> bool:
    """Read global Rasa configuration.

    Args:
        name: Name of the configuration key
        value: Value the configuration key should be set to

    Returns:
        `True` if the operation was successful.
    """
    # need to use `rasa.constants.GLOBAL_USER_CONFIG_PATH` to allow patching
    # in tests
    config_path = rasa.constants.GLOBAL_USER_CONFIG_PATH
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        c = read_global_config(config_path)
        c[name] = value
        rasa.shared.utils.io.write_yaml(c, rasa.constants.GLOBAL_USER_CONFIG_PATH)
        return True
    except Exception as e:
        logger.warning(f"Failed to write global config. Error: {e}. Skipping.")
        return False


def read_global_config_value(name: Text, unavailable_ok: bool = True) -> Any:
    """Read a value from the global Rasa configuration."""

    def not_found() -> None:
        if unavailable_ok:
            return None
        else:
            raise ValueError(f"Configuration '{name}' key not found.")

    # need to use `rasa.constants.GLOBAL_USER_CONFIG_PATH` to allow patching
    # in tests
    config_path = rasa.constants.GLOBAL_USER_CONFIG_PATH

    if not os.path.exists(config_path):
        return not_found()

    c = read_global_config(config_path)

    if name in c:
        return c[name]
    else:
        return not_found()


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


class RepeatedLogFilter(logging.Filter):
    """Filter repeated log records."""

    last_log = None

    def filter(self, record: logging.LogRecord) -> bool:
        """Determines whether current log is different to last log."""
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


def run_in_loop(
    f: Coroutine[Any, Any, T], loop: Optional[asyncio.AbstractEventLoop] = None
) -> T:
    """Execute the awaitable in the passed loop.

    If no loop is passed, the currently existing one is used or a new one is created
    if no loop has been started in the current context.

    After the awaitable is finished, all remaining tasks on the loop will be
    awaited as well (background tasks).

    WARNING: don't use this if there are never ending background tasks scheduled.
        in this case, this function will never return.

    Args:
       f: function to execute
       loop: loop to use for the execution

    Returns:
        return value from the function
    """

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    result = loop.run_until_complete(f)

    # Let's also finish all running tasks:
    # fallback for python 3.6, which doesn't have asyncio.all_tasks
    try:
        pending = asyncio.all_tasks(loop)
    except AttributeError:
        pending = asyncio.Task.all_tasks()
    loop.run_until_complete(asyncio.gather(*pending))

    return result


async def call_potential_coroutine(
    coroutine_or_return_value: Union[Any, Coroutine]
) -> Any:
    """Awaits coroutine or returns value directly if it's not a coroutine.

    Args:
        coroutine_or_return_value: Either the return value of a synchronous function
            call or a coroutine which needs to be await first.

    Returns:
        The return value of the function.
    """
    if asyncio.iscoroutine(coroutine_or_return_value):
        return await coroutine_or_return_value

    return coroutine_or_return_value
