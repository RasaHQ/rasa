#!/usr/bin/env python3
"""Main entry point for the prompt-to-bot service."""

import asyncio
import logging
import os
import sys
from typing import Optional

import structlog
from sanic import HTTPResponse, Sanic
from sanic.request import Request
from sanic_openapi import openapi3_blueprint

import rasa.core.utils
import rasa.telemetry
from rasa.builder import config
from rasa.builder.logging_utils import (
    attach_request_id_processor,
    collecting_logs_processor,
    log_request_end,
    log_request_start,
)
from rasa.builder.service import bp, setup_project_generator
from rasa.core.agent import Agent, load_agent
from rasa.core.available_endpoints import AvailableEndpoints
from rasa.core.channels.studio_chat import StudioChatInput
from rasa.model import get_latest_model
from rasa.server import configure_cors
from rasa.utils.common import configure_logging_and_warnings
from rasa.utils.log_utils import configure_structlog
from rasa.utils.sanic_error_handler import register_custom_sanic_error_handler

structlogger = structlog.get_logger()


def setup_logging() -> None:
    """Setup logging configuration."""
    log_level = logging.DEBUG

    configure_logging_and_warnings(
        log_level=log_level,
        logging_config_file=None,
        warn_only_once=True,
        filter_repeated_logs=True,
    )

    configure_structlog(
        log_level,
        include_time=True,
        additional_processors=[attach_request_id_processor, collecting_logs_processor],
    )


def setup_input_channel() -> StudioChatInput:
    """Setup the input channel for chat interactions."""
    studio_chat_credentials = config.get_default_credentials().get(
        StudioChatInput.name()
    )
    return StudioChatInput.from_credentials(credentials=studio_chat_credentials)


async def try_load_existing_agent(project_folder: str) -> Optional[Agent]:
    """Try to load an existing agent from the project's models directory.

    Args:
        project_folder: Path to the project folder

    Returns:
        Loaded Agent instance if successful, None otherwise
    """
    models_dir = os.path.join(project_folder, "models")

    if not os.path.exists(models_dir) or not os.path.isdir(models_dir):
        structlogger.debug("No models directory found", models_dir=models_dir)
        return None

    try:
        # Find the latest model in the models directory
        latest_model_path = get_latest_model(models_dir)
        if not latest_model_path:
            structlogger.debug(
                "No models found in models directory", models_dir=models_dir
            )
            return None

        structlogger.info(
            "Found existing model, attempting to load", model_path=latest_model_path
        )

        # Get available endpoints for agent loading
        available_endpoints = AvailableEndpoints.get_instance()

        # Load the agent
        agent = await load_agent(
            model_path=latest_model_path, endpoints=available_endpoints
        )

        if agent and agent.is_ready():
            structlogger.info(
                "Successfully loaded existing agent", model_path=latest_model_path
            )
            return agent
        else:
            structlogger.warning(
                "Agent loaded but not ready", model_path=latest_model_path
            )
            return None

    except Exception as e:
        structlogger.warning(
            "Failed to load existing agent",
            models_dir=models_dir,
            error=str(e),
            exc_info=True,
        )
        return None


def setup_middleware(app: Sanic) -> None:
    """Setup middleware for request/response processing."""

    @app.middleware("request")  # type: ignore[misc,no-untyped-call]
    async def log_request(request: Request) -> None:
        # store start time on request ctx for later latency calculation
        request.ctx._start_time = log_request_start(request)

    @app.middleware("response")  # type: ignore[misc,no-untyped-call]
    async def log_response(request: Request, response: HTTPResponse) -> None:
        try:
            start = getattr(request.ctx, "_start_time", None)
            if start is None:
                # If for some reason the request middleware didn't run
                start = log_request_start(request)
            # propagate correlation id for clients
            correlation_id = getattr(request.ctx, "correlation_id", None)
            if correlation_id:
                response.headers["X-Correlation-Id"] = correlation_id
            log_request_end(request, response, start)
        except Exception:
            # avoid breaking response path
            pass


def create_app(project_folder: str) -> Sanic:
    """Create and configure the Sanic app."""
    app = Sanic("BotBuilderService")

    # Basic app configuration
    app.config.REQUEST_TIMEOUT = 60  # 1 minute timeout
    # Expose auth toggle to app.config so decorators can read it
    app.config.USE_AUTHENTICATION = True

    structlogger.debug(
        "builder.main.create_app",
        project_folder=project_folder,
        use_authentication=app.config.USE_AUTHENTICATION,
        rasa_version=rasa.__version__,
    )
    app.ctx.agent = None

    # Set up project generator and store in app context
    app.ctx.project_generator = setup_project_generator(project_folder)

    # Set up input channel and store in app context
    app.ctx.input_channel = setup_input_channel()

    # Register the blueprint
    app.blueprint(bp)

    # OpenAPI docs
    app.blueprint(openapi3_blueprint)
    app.config.API_TITLE = "Bot Builder API"
    app.config.API_VERSION = rasa.__version__
    app.config.API_DESCRIPTION = (
        "API for building conversational AI bots from prompts and templates. "
        "The API allows to change the assistant and retrain it with new data."
    )

    # Setup middleware
    setup_middleware(app)

    configure_cors(app, cors_origins=config.CORS_ORIGINS)

    # Register input channel webhooks
    from rasa.core import channels

    channels.channel.register([app.ctx.input_channel], app, route="/webhooks/")

    # Register startup event handler for agent loading
    @app.after_server_start
    async def load_agent_on_startup(
        app: Sanic, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Load existing agent if available when server starts."""
        try:
            existing_agent = await try_load_existing_agent(project_folder)
            if existing_agent:
                app.ctx.agent = existing_agent
                structlogger.info("Agent loaded on server startup")
            else:
                structlogger.info(
                    "No existing agent found, server starting without agent"
                )
        except Exception as e:
            structlogger.warning("Failed to load agent on server startup", error=str(e))

    return app


def main(project_folder: Optional[str] = None) -> None:
    """Main entry point."""
    try:
        # Setup logging
        setup_logging()

        # Setup telemetry
        rasa.telemetry.initialize_telemetry()
        rasa.telemetry.initialize_error_reporting(private_mode=False)

        # working directory needs to be the project folder, e.g.
        # for relative paths (./docs) in a projects config to work
        if not project_folder:
            import tempfile

            project_folder = tempfile.mkdtemp(prefix="rasa_builder_")

        os.chdir(project_folder)

        # Create and configure app
        app = create_app(project_folder)
        register_custom_sanic_error_handler(app)

        # Run the service
        structlogger.info(
            "service.starting",
            host=config.BUILDER_SERVER_HOST,
            port=config.BUILDER_SERVER_PORT,
        )

        app.run(
            host=config.BUILDER_SERVER_HOST,
            port=config.BUILDER_SERVER_PORT,
            legacy=True,
            motd=False,
        )

    except KeyboardInterrupt:
        print("\nService stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to start service: {e}")
        sys.exit(1)


if __name__ == "__main__":
    project_folder = sys.argv[1] if len(sys.argv) > 1 else None
    main(project_folder)
