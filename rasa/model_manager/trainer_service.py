import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from pydantic import BaseModel, ConfigDict

from rasa.constants import MODEL_ARCHIVE_EXTENSION, RASA_DIR_NAME
from rasa.model_manager import config
from rasa.model_manager.utils import (
    ensure_base_directory_exists,
    logs_path,
    models_base_path,
    write_encoded_data_to_file,
)
from rasa.model_manager.warm_rasa_process import (
    start_rasa_process,
)
from rasa.model_training import generate_random_model_name
from rasa.studio.prompts import handle_prompts
from rasa.utils.io import subpath

structlogger = structlog.get_logger()


class TrainingSessionStatus(str, Enum):
    """Enum for the training status."""

    RUNNING = "running"
    STOPPED = "stopped"
    DONE = "done"
    ERROR = "error"


class TrainingSession(BaseModel):
    """Store information about a training session."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    training_id: str
    assistant_id: str
    client_id: Optional[str]
    progress: int
    model_name: str
    status: TrainingSessionStatus
    process: subprocess.Popen
    log_id: str

    def is_status_indicating_alive(self) -> bool:
        """Check if the training is running."""
        return self.status == TrainingSessionStatus.RUNNING

    def has_just_finished(self) -> bool:
        if not self.is_status_indicating_alive():
            # skip if the training is not running
            return False
        if self.process.poll() is None:
            # process is still running
            return False
        return True

    def model_path(self) -> str:
        """Return the path to the model."""
        return subpath(models_base_path(), f"{self.model_name}.tar.gz")


def train_path(training_id: str) -> str:
    """Return the path to the training directory for a given training id."""
    return subpath(config.SERVER_BASE_WORKING_DIRECTORY + "/trainings", training_id)


def cache_for_assistant_path(assistant_id: str) -> str:
    """Return the path to the cache directory for a given assistant id."""
    return subpath(config.SERVER_BASE_WORKING_DIRECTORY + "/caches", assistant_id)


def terminate_training(training: TrainingSession) -> None:
    if not training.is_status_indicating_alive():
        # no-op if the training is not running
        return

    structlogger.info(
        "model_trainer.user_stopping_training", training_id=training.training_id
    )
    try:
        training.process.terminate()
        training.status = TrainingSessionStatus.STOPPED
    except ProcessLookupError:
        structlogger.debug(
            "model_trainer.training_process_not_found",
            training_id=training.training_id,
        )
    finally:
        clean_up_after_training(training)


def update_training_status(training: TrainingSession) -> None:
    if training.has_just_finished():
        complete_training(training)


def complete_training(training: TrainingSession) -> None:
    """Complete a training session.

    Transitions the status of a training process to "done" if the process has
    finished successfully, and to "error" if the process has finished with an
    error.
    """
    if training.process.returncode == 0:
        training.status = TrainingSessionStatus.DONE
    else:
        training.status = TrainingSessionStatus.ERROR

    training.progress = 100

    structlogger.info(
        "model_trainer.training_finished",
        training_id=training.training_id,
        status=training.status,
    )

    # persist the assistant cache to speed up future training runs for this
    # assistant
    persist_rasa_cache(training.assistant_id, train_path(training.training_id))
    move_model_to_local_storage(training)

    clean_up_after_training(training)


def clean_up_after_training(training: TrainingSession) -> None:
    """Clean up the training directory."""
    structlogger.debug(
        "model_trainer.cleaning_up_training", training_id=training.training_id
    )
    shutil.rmtree(train_path(training.training_id), ignore_errors=True)


def move_model_to_local_storage(training: TrainingSession) -> None:
    """Persist the model to the remote storage."""
    ensure_base_directory_exists(models_base_path())

    model_path = subpath(
        train_path(training.training_id) + "/models",
        f"{training.model_name}.{MODEL_ARCHIVE_EXTENSION}",
    )

    if os.path.exists(model_path):
        structlogger.debug(
            "model_trainer.persisting_model_to_models_dir",
            training_model_path=model_path,
            storage_model_path=models_base_path(),
        )
        shutil.move(model_path, models_base_path())
    else:
        structlogger.warning(
            "model_trainer.model_not_found_after_training",
            training_id=training.training_id,
            model_path=model_path,
        )


def seed_training_directory_with_rasa_cache(
    training_base_path: str, assistant_id: str
) -> None:
    """Populate the training directory with the cache of a previous training."""
    # check if there is a cache for this assistant
    cache_path = cache_for_assistant_path(assistant_id)

    if os.path.exists(cache_path):
        structlogger.debug(
            "model_trainer.populating_training_dir_with_cache",
            assistant_id=assistant_id,
            training_base_path=training_base_path,
        )
        # copy the cache to the training directory
        shutil.copytree(src=cache_path, dst=subpath(training_base_path, RASA_DIR_NAME))


def persist_rasa_cache(assistant_id: str, training_base_path: str) -> None:
    """Persist the cache of a training session to speed up future trainings."""
    # copy the cache from the training directory to the cache directory
    # cache files are stored inside of `/.rasa/` of the training folder
    structlogger.debug(
        "model_trainer.persisting_assistant_cache", assistant_id=assistant_id
    )
    cache_path = cache_for_assistant_path(assistant_id)

    # if the training failed and didn't create a cache, skip this step
    if not os.path.exists(subpath(training_base_path, RASA_DIR_NAME)):
        return

    # clean up the cache directory first
    shutil.rmtree(cache_path, ignore_errors=True)
    shutil.copytree(src=subpath(training_base_path, RASA_DIR_NAME), dst=cache_path)


def write_training_data_to_files(
    encoded_training_data: Dict[str, Any], training_base_path: str
) -> None:
    """Write the training data to files in the training directory.

    Incoming data format, all keys being optional:
    ````
    {
        "domain": "base64 encoded domain.yml",
        "credentials": "base64 encoded credentials.yml",
        "endpoints": "base64 encoded endpoints.yml",
        "flows": "base64 encoded flows.yml",
        "config": "base64 encoded config.yml",
        "stories": "base64 encoded stories.yml",
        "rules": "base64 encoded rules.yml",
        "nlu": "base64 encoded nlu.yml"
        "prompts": "dictionary with the prompts",
    }
    ```
    """
    data_to_be_written_to_files = {
        "domain": "domain.yml",
        "credentials": "credentials.yml",
        "endpoints": "endpoints.yml",
        "flows": "data/flows.yml",
        "config": "config.yml",
        "stories": "data/stories.yml",
        "rules": "data/rules.yml",
        "nlu": "data/nlu.yml",
    }

    for key, file in data_to_be_written_to_files.items():
        parent_path, file_name = os.path.split(file)

        write_encoded_data_to_file(
            encoded_training_data.get(key, ""),
            subpath(training_base_path + "/" + parent_path, file_name),
        )

    if prompts := encoded_training_data.get("prompts"):
        handle_prompts(prompts, Path(training_base_path))


def prepare_training_directory(
    training_base_path: str, assistant_id: str, encoded_training_data: Dict[str, Any]
) -> None:
    """Prepare the training directory for a new training session."""
    # create a new working directory and store the training data from the
    # request there. the training data in the request is base64 encoded
    os.makedirs(training_base_path, exist_ok=True)

    seed_training_directory_with_rasa_cache(training_base_path, assistant_id)
    write_training_data_to_files(encoded_training_data, training_base_path)
    structlogger.debug("model_trainer.prepared_training", path=training_base_path)


def start_training_process(
    training_id: str,
    assistant_id: str,
    client_id: str,
    training_base_path: str,
) -> TrainingSession:
    model_name = generate_random_model_name()
    # Start the training in a subprocess
    # set the working directory to the training directory
    # run the rasa train command as a subprocess, activating poetry before running
    # pipe the stdout and stderr to the same file
    arguments = [
        "train",
        "--debug",
        "--data",
        "data",
        "--config",
        "config.yml",
        "--domain",
        "domain.yml",
        "--endpoints",
        "endpoints.yml",
        "--fixed-model-name",
        f"{model_name}.{MODEL_ARCHIVE_EXTENSION}",
        "--out",
        "models",
    ]

    if config.SERVER_MODEL_REMOTE_STORAGE:
        arguments.extend(
            [
                "--keep-local-model-copy",
                "--remote-storage",
                config.SERVER_MODEL_REMOTE_STORAGE,
                "--remote-root-only",
            ]
        )

    structlogger.debug(
        "model_trainer.training_arguments", arguments=" ".join(arguments)
    )

    warm_process = start_rasa_process(cwd=training_base_path, arguments=arguments)

    structlogger.info(
        "model_trainer.training_started",
        training_id=training_id,
        assistant_id=assistant_id,
        model_name=model_name,
        client_id=client_id,
        log=logs_path(warm_process.log_id),
        pid=warm_process.process.pid,
    )

    return TrainingSession(
        training_id=training_id,
        assistant_id=assistant_id,
        client_id=client_id,
        model_name=model_name,
        progress=0,
        status=TrainingSessionStatus.RUNNING,
        process=warm_process.process,  # Store the process handle
        log_id=warm_process.log_id,
    )


def run_training(
    training_id: str,
    assistant_id: str,
    client_id: str,
    encoded_training_data: Dict,
) -> TrainingSession:
    """Run a training session."""
    training_base_path = train_path(training_id)

    prepare_training_directory(training_base_path, assistant_id, encoded_training_data)
    return start_training_process(
        training_id=training_id,
        assistant_id=assistant_id,
        client_id=client_id,
        training_base_path=training_base_path,
    )
