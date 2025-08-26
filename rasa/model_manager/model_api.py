import asyncio
import os
from functools import wraps
from http import HTTPStatus
from typing import Any, Callable, Dict, Optional, Union

import dotenv
import psutil
import structlog
from ruamel.yaml import YAMLError
from sanic import Blueprint, Sanic, response
from sanic.exceptions import NotFound
from sanic.request import Request
from sanic.response import json
from socketio import AsyncServer

import rasa
from rasa.cli.project_templates.defaults import get_rasa_defaults
from rasa.cli.scaffold import ProjectTemplateName, scaffold_path
from rasa.constants import MODEL_ARCHIVE_EXTENSION
from rasa.exceptions import ModelNotFound
from rasa.model_manager import config
from rasa.model_manager.config import SERVER_BASE_URL
from rasa.model_manager.runner_service import (
    BotSession,
    BotSessionStatus,
    fetch_remote_model_to_dir,
    fetch_size_of_remote_model,
    run_bot,
    terminate_bot,
    update_bot_status,
)
from rasa.model_manager.socket_bridge import create_bridge_server
from rasa.model_manager.trainer_service import (
    TrainingSession,
    TrainingSessionStatus,
    run_training,
    terminate_training,
    update_training_status,
)
from rasa.model_manager.utils import (
    get_logs_content,
    logs_base_path,
    models_base_path,
)
from rasa.model_manager.warm_rasa_process import (
    initialize_warm_rasa_process,
    shutdown_warm_rasa_processes,
)
from rasa.server import ErrorResponse
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.utils.yaml import dump_obj_as_yaml_to_string
from rasa.studio.upload import build_calm_import_parts
from rasa.utils.io import InvalidPathException, subpath

dotenv.load_dotenv()

structlogger = structlog.get_logger()


# A simple in-memory store for training sessions and running bots
trainings: Dict[str, TrainingSession] = {}
running_bots: Dict[str, BotSession] = {}


def prepare_working_directories() -> None:
    """Make sure all required directories exist."""
    os.makedirs(logs_base_path(), exist_ok=True)
    os.makedirs(models_base_path(), exist_ok=True)


def cleanup_training_processes() -> None:
    """Terminate all running training processes."""
    structlogger.debug("model_trainer.cleanup_processes.started")
    running = list(trainings.values())
    for training in running:
        terminate_training(training)


def cleanup_bot_processes() -> None:
    """Terminate all running bot processes."""
    structlogger.debug("model_runner.cleanup_processes.started")
    running = list(running_bots.values())
    for bot in running:
        terminate_bot(bot)


def update_status_of_all_trainings() -> None:
    """Update the status of all training processes."""
    running = list(trainings.values())
    for training in running:
        update_training_status(training)


async def update_status_of_all_bots() -> None:
    """Update the status of all bot processes."""
    # we need to get the values first, because (since we are async and waiting
    # within the loop) some other job on the asyncio loop could change the dict
    # (adding or removing). python doesn't like if you change the size of a dict
    # while iterating over it and will raise a RuntimeError. so we get the values
    # first and iterate over them to avoid that.
    running = list(running_bots.values())
    for bot in running:
        await update_bot_status(bot)


def base_server_url(request: Request) -> str:
    """Return the base URL of the server."""
    if SERVER_BASE_URL:
        return SERVER_BASE_URL.rstrip("/")
    else:
        return f"{request.scheme}://{request.host}/{config.DEFAULT_SERVER_PATH_PREFIX}"


async def continuously_update_process_status() -> None:
    """Regularly Update the status of all training and bot processes."""
    structlogger.debug("model_api.update_process_status.started")

    while True:
        try:
            update_status_of_all_trainings()
            await update_status_of_all_bots()
        except asyncio.exceptions.CancelledError:
            structlogger.debug("model_api.update_process_status.cancelled")
            break
        except Exception as e:
            structlogger.error("model_api.update_process_status.error", error=str(e))
        finally:
            await asyncio.sleep(0.1)


def internal_blueprint() -> Blueprint:
    """Create a blueprint for the model manager API."""
    bp = Blueprint("model_api_internal")

    @bp.before_server_stop
    async def cleanup_processes(app: Sanic, loop: asyncio.AbstractEventLoop) -> None:
        """Terminate all running processes before the server stops."""
        structlogger.debug("model_api.cleanup_processes.started")
        cleanup_training_processes()
        cleanup_bot_processes()
        shutdown_warm_rasa_processes()

    @bp.after_server_start
    async def create_warm_rasa_processes(
        app: Sanic, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Create warm Rasa processes to speed up future training and bot runs."""
        structlogger.debug("model_api.create_warm_rasa_processes.started")
        initialize_warm_rasa_process()

    def limit_parallel_training_requests() -> Callable[[Callable], Callable[..., Any]]:
        """Limit the number of parallel training requests."""

        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated(*args: Any, **kwargs: Any) -> Any:
                running_requests = len(
                    [
                        training
                        for training in trainings.values()
                        if training.status == TrainingSessionStatus.RUNNING
                        and training.process.poll() is None
                    ]
                )

                if running_requests >= int(config.MAX_PARALLEL_TRAININGS):
                    return response.json(
                        {
                            "message": f"Too many parallel training requests, above "
                            f"the limit of {config.MAX_PARALLEL_TRAININGS}. "
                            f"Retry later or increase your server's "
                            f"memory and CPU resources."
                        },
                        status=HTTPStatus.TOO_MANY_REQUESTS,
                    )
                return f(*args, **kwargs)

            return decorated

        return decorator

    def limit_parallel_bot_runs() -> Callable[[Callable], Callable[..., Any]]:
        """Limit the number of parallel training requests."""

        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated(*args: Any, **kwargs: Any) -> Any:
                running_requests = len(
                    [
                        bot
                        for bot in running_bots.values()
                        if bot.status
                        in {BotSessionStatus.RUNNING, BotSessionStatus.QUEUED}
                    ]
                )

                if running_requests >= int(config.MAX_PARALLEL_BOT_RUNS):
                    return response.json(
                        {
                            "message": f"Too many parallel bot runs, above "
                            f"the limit of {config.MAX_PARALLEL_BOT_RUNS}. "
                            f"Retry later or increase your server's "
                            f"memory and CPU resources."
                        },
                        status=HTTPStatus.TOO_MANY_REQUESTS,
                    )

                return f(*args, **kwargs)

            return decorated

        return decorator

    def ensure_minimum_disk_space() -> Callable[[Callable], Callable[..., Any]]:
        """Ensure that there is enough disk space before starting a new process."""
        min_required_disk_space = 1024 * 1024 * config.MIN_REQUIRED_DISCSPACE_MB

        def decorator(f: Callable) -> Callable:
            @wraps(f)
            def decorated(*args: Any, **kwargs: Any) -> Any:
                if os.path.exists(config.SERVER_BASE_WORKING_DIRECTORY):
                    free_space_bytes = psutil.disk_usage(
                        config.SERVER_BASE_WORKING_DIRECTORY
                    ).free
                    structlogger.debug(
                        "model_api.storage.available_disk_space",
                        available_space_mb=free_space_bytes / 1024 / 1024,
                    )

                    if free_space_bytes < min_required_disk_space:
                        return response.json(
                            {
                                "message": (
                                    f"Less than {config.MIN_REQUIRED_DISCSPACE_MB} MB "
                                    f"of free disk space available. "
                                    f"Please free up some space on the model service."
                                )
                            },
                            status=HTTPStatus.INSUFFICIENT_STORAGE,
                        )

                return f(*args, **kwargs)

            return decorated

        return decorator

    @bp.get("/")
    async def health(request: Request) -> response.HTTPResponse:
        return json(
            {
                "status": "ok",
                "bots": [
                    {
                        "deployment_id": bot.deployment_id,
                        "status": bot.status,
                        "internal_url": bot.internal_url,
                        "returncode": bot.returncode,
                        "url": bot.url,
                    }
                    for bot in running_bots.values()
                ],
                "trainings": [
                    {
                        "training_id": training.training_id,
                        "assistant_id": training.assistant_id,
                        "client_id": training.client_id,
                        "progress": training.progress,
                        "status": training.status,
                    }
                    for training in trainings.values()
                ],
            }
        )

    @bp.get("/training")
    async def get_training_list(request: Request) -> response.HTTPResponse:
        """Return a list of all training sessions for an assistant."""
        assistant_id = request.args.get("assistant_id")
        sessions = [
            {
                "training_id": session.training_id,
                "assistant_id": session.assistant_id,
                "client_id": session.client_id,
                "progress": session.progress,
                "status": session.status,
                "model_name": session.model_name,
                "runtime_metadata": None,
            }
            for session in trainings.values()
            if session.assistant_id == assistant_id
        ]
        return json({"training_sessions": sessions, "total_number": len(sessions)})

    @bp.post("/training")
    @limit_parallel_training_requests()
    @ensure_minimum_disk_space()
    async def start_training(request: Request) -> response.HTTPResponse:
        """Start a new training session."""
        data = request.json
        training_id: Optional[str] = data.get("id")
        assistant_id: Optional[str] = data.get("assistant_id")
        client_id: Optional[str] = data.get("client_id")
        encoded_training_data: Dict[str, str] = data.get("bot_config", {}).get(
            "data", {}
        )

        if training_id in trainings:
            # fail, because there apparently is already a training with this id
            return json({"message": "Training with this id already exists"}, status=409)

        if not assistant_id:
            return json({"message": "Assistant id is required"}, status=400)

        if not training_id:
            return json({"message": "Training id is required"}, status=400)

        try:
            # file deepcode ignore PT: path traversal is prevented
            # by the `subpath` function found in the `rasa.model_manager.utils` module
            training_session = run_training(
                training_id=training_id,
                assistant_id=assistant_id,
                client_id=client_id,
                encoded_training_data=encoded_training_data,
            )
            trainings[training_id] = training_session
            return json(
                {"training_id": training_id, "model_name": training_session.model_name}
            )
        except InvalidPathException as exc:
            return json({"message": str(exc)}, status=403)
        except Exception as exc:
            return json({"message": str(exc)}, status=500)

    @bp.get("/training/<training_id>")
    async def get_training(request: Request, training_id: str) -> response.HTTPResponse:
        """Return the status of a training session."""
        if training := trainings.get(training_id):
            return json(
                {
                    "training_id": training_id,
                    "assistant_id": training.assistant_id,
                    "client_id": training.client_id,
                    "progress": training.progress,
                    "model_name": training.model_name,
                    "status": training.status,
                    "logs": get_logs_content(training.log_id),
                }
            )
        else:
            return json({"message": "Training not found"}, status=404)

    @bp.delete("/training/<training_id>")
    async def stop_training(
        request: Request, training_id: str
    ) -> response.HTTPResponse:
        # this is a no-op if the training is already done
        if not (training := trainings.get(training_id)):
            return json({"message": "Training session not found"}, status=404)

        terminate_training(training)
        return json({"training_id": training_id})

    @bp.post("/bot")
    @limit_parallel_bot_runs()
    @ensure_minimum_disk_space()
    async def start_bot(request: Request) -> response.HTTPResponse:
        data = request.json
        deployment_id: Optional[str] = data.get("deployment_id")
        model_name: Optional[str] = data.get("model_name")
        encoded_configs: Dict[str, str] = data.get("bot_config", {})

        if deployment_id in running_bots:
            # fail, because there apparently is already a bot running with this id
            return json(
                {"message": "Bot with this deployment id already exists"}, status=409
            )

        if not deployment_id:
            return json({"message": "Deployment id is required"}, status=400)

        if not model_name:
            return json({"message": "Model name is required"}, status=400)

        base_url_path = base_server_url(request)
        try:
            bot_session = run_bot(
                deployment_id,
                model_name,
                base_url_path,
                encoded_configs,
            )
            running_bots[deployment_id] = bot_session
            return json(
                {
                    "deployment_id": deployment_id,
                    "status": bot_session.status,
                    "url": bot_session.url,
                }
            )
        except ModelNotFound:
            return json(
                {"message": f"Model with name '{model_name}' could not be found."},
                status=404,
            )
        except Exception as e:
            return json({"message": str(e)}, status=500)

    @bp.delete("/bot/<deployment_id>")
    async def stop_bot(request: Request, deployment_id: str) -> response.HTTPResponse:
        bot = running_bots.get(deployment_id)
        if bot is None:
            return json({"message": "Bot not found"}, status=404)

        terminate_bot(bot)

        return json(
            {"deployment_id": deployment_id, "status": bot.status, "url": bot.url}
        )

    @bp.get("/bot/<deployment_id>")
    async def get_bot(request: Request, deployment_id: str) -> response.HTTPResponse:
        bot = running_bots.get(deployment_id)
        if bot is None:
            return json({"message": "Bot not found"}, status=404)

        return json(
            {
                "deployment_id": deployment_id,
                "status": bot.status,
                "returncode": bot.returncode,
                "url": bot.url,
                "logs": get_logs_content(bot.log_id),
            }
        )

    @bp.get("/bot")
    async def list_bots(request: Request) -> response.HTTPResponse:
        bots = [
            {
                "deployment_id": bot.deployment_id,
                "status": bot.status,
                "returncode": bot.returncode,
                "url": bot.url,
            }
            for bot in running_bots.values()
        ]
        return json({"deployment_sessions": bots, "total_number": len(bots)})

    @bp.route("/models/<model_name>", methods=["GET"])
    async def send_model(
        request: Request, model_name: str
    ) -> Union[response.ResponseStream, response.HTTPResponse]:
        try:
            model_path = path_to_model(model_name)

            # get size of model file
            model_size = os.stat(model_path)

            return await response.file_stream(
                model_path, headers={"Content-Length": str(model_size.st_size)}
            )
        except NotFound:
            return json({"message": "Model not found"}, status=404)
        except ModelNotFound:
            return json({"message": "Model not found"}, status=404)

    @bp.route("/models/<model_name>", methods=["HEAD"])
    async def head_model(request: Request, model_name: str) -> response.HTTPResponse:
        try:
            model_size = size_of_model(model_name)

            structlogger.debug(
                "model_api.internal.head_model",
                model_name=model_name,
                size=model_size,
            )
            return response.raw(
                b"", status=200, headers={"Content-Length": str(model_size)}
            )
        except ModelNotFound:
            return response.raw(b"", status=404)

    @bp.post("/defaults")
    async def get_defaults(request: Request) -> response.HTTPResponse:
        """Returns the system defaults like prompts, patterns, etc."""
        body = request.json or {}
        config_yaml = body.get("config")
        if config_yaml is None:
            exc = ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Missing `config` key in request body.",
            )
            return response.json(exc.error_info, status=exc.status)

        endpoints_yaml = body.get("endpoints")
        if endpoints_yaml is None:
            exc = ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                "Missing `endpoints` key in request body.",
            )
            return response.json(exc.error_info, status=exc.status)

        try:
            defaults = get_rasa_defaults(config_yaml, endpoints_yaml)
        except (YAMLError, InvalidConfigException) as e:
            exc = ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "InitDataError",
                f"Failed to load defaults. Error: {e!s}",
            )
            return response.json(exc.error_info, status=exc.status)
        return response.json(defaults.model_dump(exclude_none=True))

    @bp.get("/project_template")
    async def get_project_template(request: Request) -> response.HTTPResponse:
        """Return initial project template data."""
        template = request.args.get("template", ProjectTemplateName.DEFAULT.value)

        try:
            template_enum = ProjectTemplateName(template)
        except ValueError:
            valid_templates = ", ".join([t.value for t in ProjectTemplateName])
            exc = ErrorResponse(
                HTTPStatus.BAD_REQUEST,
                "BadRequest",
                f"Unknown template '{template}'. Valid templates: "
                f"{valid_templates}",
            )
            return response.json(exc.error_info, status=exc.status)

        template_dir = scaffold_path(template_enum)
        if not os.path.isdir(template_dir):
            exc = ErrorResponse(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "InitDataError",
                f"Template directory '{template_dir}' not found.",
            )
            return response.json(exc.error_info, status=exc.status)

        assistant_name, parts = build_calm_import_parts(
            data_path=f"{template_dir}/data",
            domain_path=f"{template_dir}/domain",
            config_path=f"{template_dir}/config.yml",
            endpoints_path=f"{template_dir}/endpoints.yml",
            assistant_name=template_enum.value,
        )

        defaults = get_rasa_defaults(
            config_yaml=dump_obj_as_yaml_to_string(parts.config),
            endpoints_yaml=dump_obj_as_yaml_to_string(parts.endpoints),
        )
        return response.json(
            {
                **parts.model_dump(exclude_none=True),
                "assistantName": assistant_name,
                "defaults": defaults.model_dump(exclude_none=True),
                "version": rasa.__version__,
            }
        )

    return bp


def external_blueprint() -> Blueprint:
    """Create a blueprint for the model manager API."""
    from rasa.core.channels.socketio import SocketBlueprint

    sio_server = AsyncServer(async_mode="sanic", cors_allowed_origins="*")
    bp = SocketBlueprint(sio_server, "", "model_api_external")

    create_bridge_server(sio_server, running_bots)

    @bp.get("/health")
    async def health(request: Request) -> response.HTTPResponse:
        return json(
            {
                "status": "ok",
                "bots": [
                    {
                        "deployment_id": bot.deployment_id,
                        "status": bot.status,
                        "internal_url": bot.internal_url,
                        "url": bot.url,
                    }
                    for bot in running_bots.values()
                ],
                "trainings": [
                    {
                        "training_id": training.training_id,
                        "assistant_id": training.assistant_id,
                        "client_id": training.client_id,
                        "progress": training.progress,
                        "status": training.status,
                    }
                    for training in trainings.values()
                ],
            }
        )

    return bp


def size_of_model(model_name: str) -> Optional[int]:
    """Return the size of a model."""
    model_file_name = f"{model_name}.{MODEL_ARCHIVE_EXTENSION}"
    model_path = subpath(models_base_path(), model_file_name)

    if os.path.exists(model_path):
        return os.path.getsize(model_path)

    if config.SERVER_MODEL_REMOTE_STORAGE:
        structlogger.debug(
            "model_api.storage.fetching_remote_model_size",
            model_name=model_file_name,
        )
        return fetch_size_of_remote_model(
            model_file_name, config.SERVER_MODEL_REMOTE_STORAGE, model_path
        )
    raise ModelNotFound("Model not found.")


def path_to_model(model_name: str) -> Optional[str]:
    """Return the path to a local model."""
    model_file_name = f"{model_name}.{MODEL_ARCHIVE_EXTENSION}"
    model_path = subpath(models_base_path(), model_file_name)

    if os.path.exists(model_path):
        return model_path

    if config.SERVER_MODEL_REMOTE_STORAGE:
        structlogger.info(
            "model_api.storage.fetching_remote_model",
            model_name=model_file_name,
        )
        return fetch_remote_model_to_dir(
            model_file_name,
            models_base_path(),
            config.SERVER_MODEL_REMOTE_STORAGE,
        )

    raise ModelNotFound("Model not found.")
