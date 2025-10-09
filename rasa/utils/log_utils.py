from __future__ import annotations

import logging
import os
import sys
from typing import Any, List, Optional

import structlog
from structlog.dev import ConsoleRenderer
from structlog.typing import EventDict, WrappedLogger
from structlog_sentry import SentryProcessor

from rasa.shared.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOG_LEVEL_LLM,
    ENV_LOG_LEVEL,
    ENV_LOG_LEVEL_LLM,
    ENV_LOG_LEVEL_LLM_MODULE_NAMES,
)

FORCE_JSON_LOGGING = os.environ.get("FORCE_JSON_LOGGING")
ANSI_CYAN_BOLD = "\033[1;36m"
ANSI_RESET = "\033[0m"


def conditional_set_exc_info(
    logger: WrappedLogger, name: str, event_dict: EventDict
) -> EventDict:
    """Set exception info only if exception does not have suppress_stack_trace flag."""
    exc_info = event_dict.get("exc_info")
    if exc_info is not None:
        is_debug_mode = logger.isEnabledFor(logging.DEBUG)

        if (
            hasattr(exc_info, "suppress_stack_trace")
            and exc_info.suppress_stack_trace
            and not is_debug_mode
        ):
            event_dict.pop("exc_info", None)
        else:
            return structlog.dev.set_exc_info(logger, name, event_dict)
    return event_dict


class HumanConsoleRenderer(ConsoleRenderer):
    """Console renderer that outputs human-readable logs."""

    def __call__(self, logger: WrappedLogger, name: str, event_dict: EventDict) -> str:
        should_highlight = event_dict.get("highlight", False)
        terminal_width = self._get_terminal_width()

        # Use event_info as title for the log entry
        if "event_info" in event_dict:
            event_key = event_dict["event"]
            event_dict["event"] = event_dict["event_info"]
            event_dict["event_key"] = event_key
            event_dict.pop("event_info", None)

        # In case the log entry should be highlighted
        # make sure to surround the log entry with ===
        event_dict = self._highlight_log_entry(
            event_dict, terminal_width, should_highlight
        )

        # Format JSON data for better readability
        event_dict = self._format_json_data(event_dict)

        # Render the log entry first
        result = super().__call__(logger, name, event_dict)

        # ensure that newlines are properly rendered
        result = "\n".join(result.split("\\n"))

        # Add closing === if we highlighted this entry
        if should_highlight:
            result += f"\n{'=' * terminal_width}"

        return result

    def _highlight_log_entry(
        self, event_dict: EventDict, terminal_width: int, should_highlight: bool
    ) -> EventDict:
        if should_highlight:
            # Only highlight if log level is DEBUG
            # structlog passes log level as 'level' or 'levelname'
            level = event_dict.get("level", event_dict.get("levelname", "")).upper()
            if level == "DEBUG":
                # Add opening === before the event info
                if "event" in event_dict and isinstance(event_dict["event"], str):
                    event_info = event_dict["event"]
                    event_dict["event"] = (
                        f"\n{'=' * terminal_width}\n"
                        f"{ANSI_CYAN_BOLD}{event_info}{ANSI_RESET}\n"
                    )

            event_dict.pop("highlight", None)

        return event_dict

    def _format_json_data(self, event_dict: EventDict) -> EventDict:
        """Format JSON data in the event dict for better readability."""
        # Get the list of fields to format from the event dict
        fields_to_format = event_dict.get("json_formatting", [])

        if not fields_to_format:
            return event_dict

        import json

        # Format only the specified fields
        for key in fields_to_format:
            if key in event_dict:
                value = event_dict[key]

                try:
                    # Try to parse as JSON if it's a string
                    if isinstance(value, str):
                        parsed = json.loads(value)
                        # If it's a dict or list, format it nicely
                        if isinstance(parsed, (dict, list)):
                            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
                            event_dict[key] = formatted
                    elif isinstance(value, (dict, list)):
                        # Format JSON with indentation for better readability
                        formatted = json.dumps(value, indent=2, ensure_ascii=False)
                        event_dict[key] = formatted
                except (TypeError, ValueError, json.JSONDecodeError):
                    # If it's not JSON serializable or if it's not valid JSON,
                    # leave it as is
                    pass

        # Remove the json_formatting key from the output
        event_dict.pop("json_formatting", None)

        return event_dict

    def _get_terminal_width(self) -> int:
        """Get the width of the terminal."""
        import shutil

        try:
            return shutil.get_terminal_size((80, 20)).columns
        except Exception:
            return 80


def configure_structlog(
    log_level: Optional[int] = None,
    include_time: bool = False,
    additional_processors: Optional[List[structlog.typing.Processor]] = None,
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

    shared_processors: List[structlog.typing.Processor] = [
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
        conditional_set_exc_info,
        # add structlog sentry integration. only log fatal log entries
        # as events as we are tracking exceptions anyways
        SentryProcessor(event_level=logging.FATAL),
    ]

    if include_time:
        shared_processors.append(structlog.processors.TimeStamper(fmt="iso"))

    if additional_processors:
        shared_processors.extend(additional_processors)

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
        processors=processors,
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
    # doing logger creation inline, to prevent usage of unconfigured logger
    structlog.get_logger().debug("structlog.configured")


def log_llm(logger: Any, log_module: str, log_event: str, **kwargs: Any) -> None:
    """Logs LLM-specific events depending on a flag passed through an env var.

    If the module's flag is set to INFO (e.g.
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
