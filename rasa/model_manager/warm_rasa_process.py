import os
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass
from typing import List

import structlog

from rasa.__main__ import main
from rasa.dialogue_understanding.constants import (
    RASA_RECORD_COMMANDS_AND_PROMPTS_ENV_VAR_NAME,
)
from rasa.model_manager import config
from rasa.model_manager.utils import ensure_base_directory_exists, logs_path

structlogger = structlog.get_logger(__name__)

warm_rasa_processes: List["WarmRasaProcess"] = []

NUMBER_OF_INITIAL_PROCESSES = 3


@dataclass
class WarmRasaProcess:
    """Data class to store a warm Rasa process.

    A "warm" Rasa process is one where we've done the heavy lifting of
    importing key modules ahead of time (e.g. litellm). This is to avoid
    long import times when we actually want to run a command.

    This is a started process waiting for a Rasa CLI command. It's
    output is stored in a log file identified by `log_id`.
    """

    process: subprocess.Popen
    log_id: str


def _create_warm_rasa_process() -> WarmRasaProcess:
    """Create a new warm Rasa process."""
    command = [
        config.RASA_PYTHON_PATH,
        "-m",
        "rasa.model_manager.warm_rasa_process",
    ]

    envs = os.environ.copy()
    envs[RASA_RECORD_COMMANDS_AND_PROMPTS_ENV_VAR_NAME] = "true"

    log_id = uuid.uuid4().hex
    log_path = logs_path(log_id)

    ensure_base_directory_exists(log_path)

    process = subprocess.Popen(
        command,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE,
        env=envs,
    )

    structlogger.debug(
        "model_trainer.created_warm_rasa_process",
        pid=process.pid,
        command=command,
        log_path=log_path,
    )

    return WarmRasaProcess(process=process, log_id=log_id)


def initialize_warm_rasa_process() -> None:
    """Initialize the warm Rasa processes."""
    global warm_rasa_processes
    for _ in range(NUMBER_OF_INITIAL_PROCESSES):
        warm_rasa_processes.append(_create_warm_rasa_process())


def shutdown_warm_rasa_processes() -> None:
    """Shutdown all warm Rasa processes."""
    global warm_rasa_processes
    for warm_rasa_process in warm_rasa_processes:
        warm_rasa_process.process.terminate()
    warm_rasa_processes = []


def start_rasa_process(cwd: str, arguments: List[str]) -> WarmRasaProcess:
    """Start a Rasa process.

    This will start a Rasa process with the given current working directory
    and arguments. The process will be a warm one, meaning that it has already
    imported all necessary modules.
    """
    warm_rasa_process = _get_warm_rasa_process()
    _pass_arguments_to_process(warm_rasa_process.process, cwd, arguments)
    return warm_rasa_process


def _get_warm_rasa_process() -> WarmRasaProcess:
    """Get a warm Rasa process.

    This will return a warm Rasa process from the pool and create a
    new one to replace it.
    """
    global warm_rasa_processes

    if not warm_rasa_processes:
        warm_rasa_processes = [_create_warm_rasa_process()]

    previous_warm_rasa_process = warm_rasa_processes.pop(0)

    if previous_warm_rasa_process.process.poll() is not None:
        # process has finished (for some reason...)
        # back up plan is to create a new one on the spot.
        # this should not happen, but let's be safe
        structlogger.warning(
            "model_trainer.warm_rasa_process_finished_unexpectedly",
            pid=previous_warm_rasa_process.process.pid,
        )
        previous_warm_rasa_process = _create_warm_rasa_process()

    warm_rasa_processes.append(_create_warm_rasa_process())
    return previous_warm_rasa_process


def _pass_arguments_to_process(
    process: subprocess.Popen, cwd: str, arguments: List[str]
) -> None:
    """Pass arguments to a warm Rasa process.

    The process is waiting for input on stdin. We pass the current working
    directory and the arguments to run a Rasa CLI command.
    """
    arguments_string = " ".join(arguments)
    # send arguments to stdin
    process.stdin.write(cwd.encode())  # type: ignore[union-attr]
    process.stdin.write("\n".encode())  # type: ignore[union-attr]
    process.stdin.write(arguments_string.encode())  # type: ignore[union-attr]
    process.stdin.write("\n".encode())  # type: ignore[union-attr]
    process.stdin.flush()  # type: ignore[union-attr]


def warmup() -> None:
    """Import all necessary modules to warm up the process.

    This should include all the modules that take a long time to import.
    We import them now, so that the training / deployment can later
    directly start.
    """
    start_time = time.time()
    try:
        import langchain  # noqa: F401
        import litellm  # noqa: F401
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import tensorflow  # noqa: F401

        import rasa.validator  # noqa: F401
    except ImportError as e:
        structlogger.warning(
            "model_trainer.warmup_error",
            error=str(e),
        )
        pass
    finally:
        end_time = time.time()
        structlogger.debug(
            "model_trainer.warmup_time",
            time_in_seconds=end_time - start_time,
        )


def warm_rasa_main() -> None:
    """Entry point for processes waiting for their command to run.

    The process will wait for the current working directory and the command
    to run. These will be send on stdin by the parent process. After receiving
    the input, we will kick things of starting or running a bot.

    Uses the normal Rasa CLI entry point (e.g. `rasa train --data ...`).
    """
    warmup()

    cwd = input()

    # this should be `train --data ...` or similar
    cli_arguments_str = input()
    # splits the arguments string into a list of arguments as expected by `argparse`
    arguments = shlex.split(cli_arguments_str)

    # needed to make sure the passed arguments are relative to the working directory
    os.chdir(cwd)

    main(arguments)


if __name__ == "__main__":
    warm_rasa_main()
