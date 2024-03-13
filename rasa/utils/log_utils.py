from __future__ import annotations
import os
import logging
import sys
from typing import Any, Dict, Optional

import structlog
from structlog_sentry import SentryProcessor
from structlog.dev import ConsoleRenderer
from structlog.typing import EventDict, WrappedLogger
from rasa.shared.constants import (
    ENV_LOG_LEVEL,
    DEFAULT_LOG_LEVEL,
    ENV_LOG_LEVEL_LLM,
    ENV_LOG_LEVEL_LLM_MODULE_NAMES,
    DEFAULT_LOG_LEVEL_LLM,
)
from rasa.plugin import plugin_manager


FORCE_JSON_LOGGING = os.environ.get("FORCE_JSON_LOGGING")


class HumanConsoleRenderer(ConsoleRenderer):
    """Console renderer that outputs human-readable logs."""

    def __call__(self, logger: WrappedLogger, name: str, event_dict: EventDict) -> str:
        if "event_info" in event_dict:
            event_key = event_dict["event"]
            event_dict["event"] = event_dict["event_info"]
            event_dict["event_key"] = event_key
            del event_dict["event_info"]

        return super().__call__(logger, name, event_dict)


def _anonymizer(
    _: structlog.BoundLogger, __: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Anonymizes event dict."""
    anonymizable_keys = [
        "text",
        "response_text",
        "user_text",
        "slots",
        "parse_data_text",
        "parse_data_entities",
        "prediction_events",
        "tracker_latest_message",
        "prefilled_slots",
        "message",
        "response",
        "slot_candidates",
        "rasa_event",
        "rasa_events",
        "tracker_states",
        "current_states",
        "old_states",
        "current_states",
        "successes",
        "current_entity",
        "next_entity",
        "states",
        "entity",
        "token_text",
        "user_message",
        "json_message",
    ]
    anonymization_pipeline = plugin_manager().hook.get_anonymization_pipeline()

    if anonymization_pipeline:
        for key in anonymizable_keys:
            if key in event_dict:
                anonymized_value = anonymization_pipeline.log_run(event_dict[key])
                event_dict[key] = anonymized_value
    return event_dict


def configure_structlog(
    log_level: Optional[int] = None,
) -> None:
    """Configure logging of the server."""
    if log_level is None:  # Log level NOTSET is 0 so we use `is None` here
        log_level_name = os.environ.get(ENV_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        # Change log level from str to int (note that log_level in function parameter
        # int already, coming from CLI argparse parameter).
        log_level = logging.getLevelName(log_level_name)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    shared_processors = [
        _anonymizer,
        # Processors that have nothing to do with output,
        # e.g., add timestamps or log level names.
        # If log level is too low, abort pipeline and throw away log entry.
        structlog.stdlib.filter_by_level,
        structlog.contextvars.merge_contextvars,
        # Add the name of the logger to event dict.
        # structlog.stdlib.add_logger_name,
        # Add log level to event dict.
        structlog.processors.add_log_level,
        # If the "stack_info" key in the event dict is true, remove it and
        # render the current stack trace in the "stack" key.
        structlog.processors.StackInfoRenderer(),
        # If some value is in bytes, decode it to a unicode str.
        structlog.processors.UnicodeDecoder(),
        structlog.dev.set_exc_info,
        # add structlog sentry integration. only log fatal log entries
        # as events as we are tracking exceptions anyways
        SentryProcessor(event_level=logging.FATAL),
    ]

    if not FORCE_JSON_LOGGING and sys.stderr.isatty():
        # Pretty printing when we run in a terminal session.
        # Automatically prints pretty tracebacks when "rich" is installed
        processors = shared_processors + [
            HumanConsoleRenderer(),
        ]
    else:
        # Print JSON when we run, e.g., in a Docker container.
        # Also print structured tracebacks.
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,  # type: ignore
        context_class=dict,
        # `logger_factory` is used to create wrapped loggers that are used for
        # OUTPUT. This one returns a `logging.Logger`. The final value (a JSON
        # string) from the final processor (`JSONRenderer`) will be passed to
        # the method of the same name as that you've called on the bound logger.
        logger_factory=structlog.stdlib.LoggerFactory(),
        # `wrapper_class` is the bound logger that you get back from
        # get_logger(). This one imitates the API of `logging.Logger`.
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        # Effectively freeze configuration after creating the first bound
        # logger.
        cache_logger_on_first_use=True,
    )


def log_llm(logger: Any, log_module: str, log_event: str, **kwargs: Any) -> None:
    """Logs LLM-specific events depending on a flag passed through an environment
    variable. If the module's flag is set to INFO (e.g.
    LOG_PROMPT_LLM_COMMAND_GENERATOR=INFO), its prompt is logged at INFO level,
    overriding the general log level setting.

    Args:
        logger: instance of the structlogger of the component
        log_module: name of the module/component logging the event
        log_event: string describing the log event
        **kwargs: dictionary of additional logging context
    """
    log_level_llm_name = os.environ.get(ENV_LOG_LEVEL_LLM, DEFAULT_LOG_LEVEL_LLM)
    log_level_llm = logging.getLevelName(log_level_llm_name.upper())

    module_env_variable = ENV_LOG_LEVEL_LLM_MODULE_NAMES.get(
        log_module, "LOG_LEVEL_LLM_" + log_module.upper()
    )
    log_level_llm_module_name = os.environ.get(
        module_env_variable, DEFAULT_LOG_LEVEL_LLM
    )
    log_level_llm_module = logging.getLevelName(log_level_llm_module_name.upper())

    # log at the highest specified level, e.g. max(DEBUG=10, INFO=20)
    log_level = max(log_level_llm, log_level_llm_module)

    logger.log(log_level, log_event, **kwargs)
