import logging
import sys
from typing import Any, Dict

import structlog
from structlog_sentry import SentryProcessor


def _anonymizer(
    logger: structlog.BoundLogger, name: str, event_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Anonymizes event dict."""
    # TODO: Replace "anonymised text" with pipeline.log_run()
    event_dict["event"] = "anonymised text"
    return event_dict


def configure_logging() -> None:
    """Configure logging of the server."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG,
    )

    shared_processors = [
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
        structlog.processors.TimeStamper(fmt="iso"),
        # add structlog sentry integration. only log fatal log entries
        # as events as we are tracking exceptions anyways
        SentryProcessor(event_level=logging.FATAL),
        # Print JSON when we run, e.g., in a Docker container.
        # Also print structured tracebacks.
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    ]

    # TODO: put in an if pipeline block
    processors = shared_processors + [_anonymizer]

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
        wrapper_class=structlog.stdlib.BoundLogger,
        # Effectively freeze configuration after creating the first bound
        # logger.
        cache_logger_on_first_use=True,
    )
