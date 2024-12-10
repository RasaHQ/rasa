import subprocess
import sys
from rasa.__main__ import main
import os
from typing import List
import structlog
from dataclasses import dataclass
import uuid

from rasa.model_manager import config
from rasa.model_manager.utils import ensure_base_directory_exists, logs_path

structlogger = structlog.get_logger(__name__)

warm_rasa_processes: List["WarmRasaProcess"] = []


@dataclass
class WarmRasaProcess:
    process: subprocess.Popen
    log_id: str


def _create_warm_rasa_process() -> WarmRasaProcess:
    command = [
        config.RASA_PYTHON_PATH,
        "-m",
        "rasa.model_manager.warm_rasa_process",
    ]

    envs = os.environ.copy()
    envs["RASA_TELEMETRY_ENABLED"] = "false"

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
    global warm_rasa_processes
    warm_rasa_processes.append(_create_warm_rasa_process())
    warm_rasa_processes.append(_create_warm_rasa_process())
    warm_rasa_processes.append(_create_warm_rasa_process())


def get_warm_rasa_process() -> WarmRasaProcess:
    global warm_rasa_processes

    if not warm_rasa_processes:
        warm_rasa_processes = [_create_warm_rasa_process()]

    previous_warm_rasa_process = warm_rasa_processes.pop(0)
    warm_rasa_processes.append(_create_warm_rasa_process())
    return previous_warm_rasa_process


def pass_arguments_to_process(
    process: subprocess.Popen, cwd: str, arguments: List[str]
) -> None:
    arguments_string = " ".join(arguments)
    # send arguments to stdin
    process.stdin.write(cwd.encode())
    process.stdin.write("\n".encode())
    process.stdin.write(arguments_string.encode())
    process.stdin.write("\n".encode())
    process.stdin.flush()


def warmup() -> None:
    try:
        import presidio_analyzer  # noqa: F401
        import litellm  # noqa: F401
        import langchain  # noqa: F401
        import tensorflow  # noqa: F401
        import matplotlib  # noqa: F401
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        import spacy  # noqa: F401
    except ImportError:
        pass

    # programmatically import rasa and all its submodules automatically
    packages_to_import = ["rasa"]
    while packages_to_import:
        package = packages_to_import.pop(0)
        try:
            __import__(package)
            module = sys.modules[package]
            if hasattr(module, "__all__"):
                for submodule in module.__all__:
                    packages_to_import.append(f"{package}.{submodule}")
        except Exception as e:
            structlogger.error(
                "model_trainer.warmup.failed_importing_package",
                package=package,
                error=str(e),
            )
            continue


def warm_rasa_main() -> None:
    """Started in a process and waiting for CLI arguments to be send over stdin."""
    warmup()

    cwd = input()
    cli_arguments = input()

    os.chdir(cwd)

    main(cli_arguments)


if __name__ == "__main__":
    warm_rasa_main()
