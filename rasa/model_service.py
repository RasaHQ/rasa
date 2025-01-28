import logging
from urllib.parse import urlparse

import structlog
from sanic import Sanic

from rasa.cli.scaffold import print_error_and_exit
from rasa.core.persistor import RemoteStorageType, get_persistor
from rasa.core.utils import list_routes
from rasa.model_manager import config, model_api
from rasa.model_manager.config import SERVER_BASE_URL, SERVER_PORT
from rasa.utils.common import configure_logging_and_warnings
from rasa.utils.log_utils import configure_structlog
from rasa.utils.sanic_error_handler import register_custom_sanic_error_handler

structlogger = structlog.get_logger()


def url_prefix_from_base_url() -> str:
    """Return the path prefix from the base URL."""
    # return path without any trailing slashes
    prefix = urlparse(SERVER_BASE_URL).path.rstrip("/") if SERVER_BASE_URL else ""

    # can't be empty
    return prefix or config.DEFAULT_SERVER_PATH_PREFIX


def validate_model_storage_type() -> None:
    """Validate the storage type if remote storage is used for models."""
    if config.SERVER_MODEL_REMOTE_STORAGE:
        if config.SERVER_MODEL_REMOTE_STORAGE not in RemoteStorageType.list():
            print_error_and_exit(
                f"Invalid storage type '{config.SERVER_MODEL_REMOTE_STORAGE}'. "
                f"Supported storage types: {', '.join(RemoteStorageType.list())}."
                f"Alternatively, unset the remote storage type to store models locally."
            )
        else:
            structlogger.info(
                "model_api.storage.remote_storage_enabled",
                remote_storage=config.SERVER_MODEL_REMOTE_STORAGE,
            )
        # try to create a client to validate the configuration
        get_persistor(config.SERVER_MODEL_REMOTE_STORAGE)
    else:
        structlogger.info(
            "model_api.storage.local_storage_enabled",
            base_path=config.SERVER_BASE_WORKING_DIRECTORY,
        )


def _register_update_task(app: Sanic) -> None:
    app.add_task(
        model_api.continuously_update_process_status,
        name="continuously_update_process_status",
    )


def main() -> None:
    """Start the Rasa Model Manager server.

    The API server can receive requests to train models, run bots, and manage
    the lifecycle of models and bots.
    """
    import rasa.utils.licensing

    log_level = logging.DEBUG
    configure_logging_and_warnings(
        log_level=log_level,
        logging_config_file=None,
        warn_only_once=True,
        filter_repeated_logs=True,
    )
    configure_structlog(log_level, include_time=True)

    rasa.utils.licensing.validate_license_from_env()

    try:
        model_api.prepare_working_directories()
    except Exception as e:
        structlogger.error(
            "model_api.prepare_directories.failed",
            error=str(e),
            base_directory=config.SERVER_BASE_WORKING_DIRECTORY,
        )
        print_error_and_exit(
            f"Failed to create working directories. Please make sure the "
            f"server base directory at '{config.SERVER_BASE_WORKING_DIRECTORY}' "
            f"is writable by the current user."
        )

    validate_model_storage_type()

    structlogger.debug("model_api.starting_server", port=SERVER_PORT)

    url_prefix = url_prefix_from_base_url()
    # configure the sanic application
    app = Sanic("RasaModelService")
    app.after_server_start(_register_update_task)
    app.blueprint(model_api.external_blueprint(), url_prefix=url_prefix)
    app.blueprint(model_api.internal_blueprint())

    # list all routes
    list_routes(app)

    register_custom_sanic_error_handler(app)

    app.run(host="0.0.0.0", port=SERVER_PORT, legacy=True, motd=False)


if __name__ == "__main__":
    main()
