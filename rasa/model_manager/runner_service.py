import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import aiohttp
import structlog
from pydantic import BaseModel, ConfigDict

from rasa.constants import MODEL_ARCHIVE_EXTENSION
from rasa.exceptions import ModelNotFound
from rasa.model_manager import config
from rasa.model_manager.utils import (
    logs_path,
    models_base_path,
    write_encoded_data_to_file,
)
from rasa.model_manager.warm_rasa_process import start_rasa_process
from rasa.studio.prompts import handle_prompts
from rasa.utils.io import subpath

structlogger = structlog.get_logger()


class BotSessionStatus(str, Enum):
    """Enum for the bot status."""

    QUEUED = "queued"
    RUNNING = "running"
    STOPPED = "stopped"


class BotSession(BaseModel):
    """Store information about a running bot."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deployment_id: str
    status: BotSessionStatus
    process: subprocess.Popen
    url: str
    internal_url: str
    port: int
    log_id: str
    returncode: Optional[int] = None

    def is_alive(self) -> bool:
        """Check if the bot is alive."""
        return self.process.poll() is None

    def is_status_indicating_alive(self) -> bool:
        """Check if the status indicates that the bot is alive."""
        return self.status in {BotSessionStatus.QUEUED, BotSessionStatus.RUNNING}

    def has_died_recently(self) -> bool:
        """Check if the bot has died recently.

        Process will indicate that the bot exited,
        but status is not yet updated.
        """
        return self.is_status_indicating_alive() and not self.is_alive()

    async def completed_startup_recently(self) -> bool:
        """Check if the bot has completed startup recently."""
        return self.status == BotSessionStatus.QUEUED and await is_bot_startup_finished(
            self
        )


def bot_path(deployment_id: str) -> str:
    """Return the path to the bot directory for a given deployment id."""
    return os.path.abspath(
        f"{config.SERVER_BASE_WORKING_DIRECTORY}/bots/{deployment_id}"
    )


async def is_bot_startup_finished(bot: BotSession) -> bool:
    """Send a request to the bot to see if the bot is up and running."""
    health_timeout = aiohttp.ClientTimeout(total=5, sock_connect=2, sock_read=3)
    try:
        async with aiohttp.ClientSession(timeout=health_timeout) as session:
            # can't use /status as by default the bot API is not enabled, only
            # the input channel
            async with session.get(f"{bot.internal_url}/license") as resp:
                return resp.status == 200
    except aiohttp.client_exceptions.ClientConnectorError:
        return False


def set_bot_status_to_stopped(bot: BotSession) -> None:
    """Set a bots state to stopped."""
    structlogger.info(
        "model_runner.bot_stopped",
        deployment_id=bot.deployment_id,
        status=bot.process.returncode,
    )
    bot.status = BotSessionStatus.STOPPED
    bot.returncode = bot.process.returncode


def set_bot_status_to_running(bot: BotSession) -> None:
    """Set a bots state to running."""
    structlogger.info(
        "model_runner.bot_running",
        deployment_id=bot.deployment_id,
    )
    bot.status = BotSessionStatus.RUNNING


def get_open_port() -> int:
    """Get an open port on the system that is not in use yet."""
    # from https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python/2838309#2838309
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def write_encoded_config_data_to_files(
    encoded_configs: Dict[str, Union[bytes, Dict[str, str]]], base_path: str
) -> None:
    """Write the encoded config data to files."""
    endpoints_encoded = encoded_configs.get("endpoints")
    if endpoints_encoded:
        write_encoded_data_to_file(
            endpoints_encoded, subpath(base_path, "endpoints.yml")
        )
    config_encoded = encoded_configs.get("config")
    if config_encoded:
        write_encoded_data_to_file(config_encoded, subpath(base_path, "config.yml"))
    credentials_encoded = encoded_configs.get("credentials")
    if credentials_encoded:
        write_encoded_data_to_file(
            credentials_encoded, subpath(base_path, "credentials.yml")
        )

    if prompts := encoded_configs.get("prompts"):
        handle_prompts(prompts, Path(base_path))


def prepare_bot_directory(
    bot_base_path: str,
    model_name: str,
    encoded_configs: Dict[str, str],
) -> None:
    """Prepare the bot directory for a new bot session."""
    if not os.path.exists(bot_base_path):
        os.makedirs(bot_base_path, exist_ok=True)
    else:
        shutil.rmtree(bot_base_path, ignore_errors=True)

    model_file_name = f"{model_name}.{MODEL_ARCHIVE_EXTENSION}"
    model_path = subpath(models_base_path(), model_file_name)

    if config.SERVER_MODEL_REMOTE_STORAGE and not os.path.exists(model_path):
        fetch_remote_model_to_dir(
            model_file_name,
            models_base_path(),
            config.SERVER_MODEL_REMOTE_STORAGE,
        )

    if not os.path.exists(model_path):
        raise ModelNotFound(f"Model '{model_file_name}' not found in '{model_path}'.")

    os.makedirs(subpath(bot_base_path, "models"), exist_ok=True)
    shutil.copy(
        src=model_path,
        dst=subpath(bot_base_path, "models"),
    )

    write_encoded_config_data_to_files(encoded_configs, bot_base_path)


def fetch_remote_model_to_dir(
    model_name: str, target_path: str, storage_type: str
) -> str:
    """Fetch the model from remote storage.

    Returns the path to the model directory.
    """
    from rasa.core.persistor import get_persistor

    persistor = get_persistor(storage_type)

    # we know there must be a persistor, because the config is set
    # this is here to please the type checker for the call below
    assert persistor is not None

    try:
        return persistor.retrieve(model_name=model_name, target_path=target_path)
    except FileNotFoundError as e:
        raise ModelNotFound("Model not found") from e


def fetch_size_of_remote_model(
    model_name: str, storage_type: str, model_path: str
) -> int:
    """Fetch the size of the model from remote storage."""
    from rasa.core.persistor import get_persistor

    persistor = get_persistor(storage_type)

    # we now there must be a persistor, because the config is set
    # this is here to please the type checker for the call below
    assert persistor is not None

    return persistor.size_of_persisted_model(
        model_name=model_name, target_path=model_path
    )


def start_bot_process(
    deployment_id: str, bot_base_path: str, base_url_path: str
) -> BotSession:
    port = get_open_port()

    arguments = [
        "run",
        "--endpoints",
        f"{bot_base_path}/endpoints.yml",
        "--credentials",
        f"{bot_base_path}/credentials.yml",
        "--debug",
        f"--port={port}",
        "--model",
        f"{bot_base_path}/models",
    ]

    structlogger.debug(
        "model_runner.bot.starting_command",
        deployment_id=deployment_id,
        arguments=" ".join(arguments),
    )

    warm_process = start_rasa_process(cwd=bot_base_path, arguments=arguments)

    internal_bot_url = f"http://localhost:{port}"

    structlogger.info(
        "model_runner.bot.starting",
        deployment_id=deployment_id,
        log=logs_path(warm_process.log_id),
        url=internal_bot_url,
        port=port,
        pid=warm_process.process.pid,
    )

    return BotSession(
        deployment_id=deployment_id,
        status=BotSessionStatus.QUEUED,
        process=warm_process.process,
        url=f"{base_url_path}?deployment_id={deployment_id}",
        internal_url=internal_bot_url,
        port=port,
        log_id=warm_process.log_id,
    )


def run_bot(
    deployment_id: str,
    model_name: str,
    base_url_path: str,
    encoded_configs: Dict[str, str],
) -> BotSession:
    """Deploy a bot based on a given training id."""
    with structlog.contextvars.bound_contextvars(model_name=model_name):
        bot_base_path = bot_path(deployment_id)
        prepare_bot_directory(bot_base_path, model_name, encoded_configs)

        return start_bot_process(deployment_id, bot_base_path, base_url_path)


async def update_bot_status(bot: BotSession) -> None:
    """Update the status of a bot based on the process return code."""
    try:
        if bot.has_died_recently():
            set_bot_status_to_stopped(bot)
        elif await bot.completed_startup_recently():
            set_bot_status_to_running(bot)
    except Exception as e:
        structlogger.error("model_runner.update_bot_status.error", error=str(e))


def terminate_bot(bot: BotSession) -> None:
    """Terminate the bot process."""
    if not bot.is_status_indicating_alive():
        # if the bot is not running, we don't need to terminate it
        return

    try:
        bot.process.terminate()
        structlogger.info(
            "model_runner.stop_bot.stopped",
            deployment_id=bot.deployment_id,
            status=bot.process.returncode,
        )
        bot.status = BotSessionStatus.STOPPED
        bot.returncode = bot.process.returncode
    except ProcessLookupError:
        structlogger.debug(
            "model_runner.stop_bot.process_not_found",
            deployment_id=bot.deployment_id,
        )
