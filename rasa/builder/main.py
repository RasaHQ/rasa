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
    collecting_validation_logs_processor,
    log_request_end,
    log_request_start,
)
from rasa.builder.service import bp, setup_project_generator
from rasa.builder.training_service import try_load_existing_agent, update_agent
from rasa.core.channels.studio_chat import StudioChatInput
from rasa.model_manager.warm_rasa_process import warmup
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
        additional_processors=[
            attach_request_id_processor,
            collecting_logs_processor,
            collecting_validation_logs_processor,
        ],
    )


def setup_input_channel() -> StudioChatInput:
    """Setup the input channel for chat interactions."""
    studio_chat_credentials = config.get_default_credentials().get(
        StudioChatInput.name()
    )
    return StudioChatInput.from_credentials(credentials=studio_chat_credentials)


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

    # Set up project generator and store in app context
    app.ctx.project_generator = setup_project_generator(project_folder)

    # Set up input channel and store in app context
    app.ctx.input_channel = setup_input_channel()

    update_agent(None, app)

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
                update_agent(existing_agent, app)
                structlogger.info("Agent loaded on server startup")
            else:
                structlogger.info(
                    "No existing agent found, server starting without agent"
                )
        except Exception as e:
            structlogger.warning("Failed to load agent on server startup", error=str(e))

    return app


def _apply_llm_overrides_from_builder_env() -> None:
    # Prefer a dedicated builder key, fall back to license if you proxy with it
    if not config.HELLO_LLM_PROXY_BASE_URL:
        return

    structlogger.debug(
        "builder.main.using_llm_proxy", base_url=config.HELLO_LLM_PROXY_BASE_URL
    )

    if not config.RASA_PRO_LICENSE:
        structlogger.error(
            "copilot.proxy_missing_license",
            event_info=(
                "HELLO_LLM_PROXY_BASE_URL is set but RASA_PRO_LICENSE is missing."
            ),
        )
        return

    if not os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_API_KEY"):
        base_url = config.HELLO_LLM_PROXY_BASE_URL.rstrip("/")
        # needed for litellm client
        os.environ["OPENAI_API_BASE"] = base_url
        # needed for openai async client
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = config.RASA_PRO_LICENSE


def main(project_folder: Optional[str] = None) -> None:
    """Main entry point."""
    try:
        # Setup logging
        setup_logging()

        # Setup telemetry
        rasa.telemetry.initialize_telemetry()
        rasa.telemetry.initialize_error_reporting(private_mode=False)

        _apply_llm_overrides_from_builder_env()

        if config.HELLO_RASA_PROJECT_ID:
            # ensures long import times for modules are ahead of time
            warmup()

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
