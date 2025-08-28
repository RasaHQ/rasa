import datetime as dt
import os
import uuid
from typing import Iterable, Optional, Sequence

import structlog

from rasa import telemetry
from rasa.builder.copilot.constants import COPILOT_SEGMENT_WRITE_KEY_ENV_VAR
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.document_retrieval.models import Document
from rasa.telemetry import (
    SEGMENT_TRACK_ENDPOINT,
    segment_request_payload,
    send_segment_request,
    with_default_context_fields,
)

structlogger = structlog.get_logger()

COPILOT_USER_MESSAGE_EVENT = "copilot_user_message"
COPILOT_BOT_MESSAGE_EVENT = "copilot_bot_message"

COPILOT_SEGMENT_WRITE_KEY = os.getenv(COPILOT_SEGMENT_WRITE_KEY_ENV_VAR)
if _SEGMENT_ON := bool(COPILOT_SEGMENT_WRITE_KEY):
    structlogger.info("builder.telemetry.enabled")
else:
    structlogger.warning("builder.telemetry.disabled")


def _track(event: str, user_id: str, properties: dict) -> None:
    """Track an event with Segment.

    Args:
        event: The name of the event to track.
        user_id: The ID of the user associated with the event.
        properties: Additional properties to include with the event.

    Raises:
        Exception: If tracking fails, an exception is logged.
    """
    if not _SEGMENT_ON or not telemetry.is_telemetry_enabled():
        structlogger.debug("builder.telemetry._track.disabled")
        return
    structlogger.debug("builder.telemetry._track.enabled")

    try:
        payload = segment_request_payload(
            user_id, event, properties, context=with_default_context_fields()
        )
        structlogger.debug("builder.telemetry._track.sending", payload=payload)

        send_segment_request(SEGMENT_TRACK_ENDPOINT, payload, COPILOT_SEGMENT_WRITE_KEY)
    except Exception as e:  # skipcq:PYL-W0703
        structlogger.warning("builder.telemetry.track_failed", error=str(e))


class CopilotTelemetry:
    def __init__(
        self,
        *,
        project_id: str,
        user_id: str,
    ) -> None:
        """Initialize Telemetry instance."""
        self._project_id = project_id
        self._user_id = user_id
        # TODO Load prompt version
        self._prompt_version = "1"

    def log_user_turn(self, text: str) -> None:
        """Track a user message in the conversation.

        Args:
            text: The text of the user message.
        """
        structlogger.debug("builder.telemetry.log_user_turn", text=text)
        _track(
            COPILOT_USER_MESSAGE_EVENT,
            self._user_id,
            {
                "project_id": self._project_id,
                "message_id": uuid.uuid4().hex,
                "text": text,
                "timestamp": dt.datetime.utcnow().isoformat(),
            },
        )

    def log_copilot_turn(
        self,
        *,
        text: str,
        source_urls: Sequence[str],
        flags: Iterable[str],
        latency_ms: int,
        model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        """Track a copilot message in the conversation.

        Args:
            text: The text of the copilot message.
            source_urls: URLs of the sources used to generate the response.
            flags: Flags indicating special conditions or features.
            latency_ms: End-to-end Copilot latency to produce this response.
            model: The model used to generate the response.
            input_tokens: Number of input tokens used (optional).
            output_tokens: Number of output tokens generated (optional).
            total_tokens: Total number of tokens used (input + output) (optional).
        """
        structlogger.debug("builder.telemetry.log_copilot_turn", text=text)
        _track(
            COPILOT_BOT_MESSAGE_EVENT,
            self._user_id,
            {
                "project_id": self._project_id,
                "message_id": uuid.uuid4().hex,
                "text": text,
                "prompt_version": self._prompt_version,
                "source_urls": list(source_urls),
                "flags": list(flags),
                "latency_ms": latency_ms,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "system_prompt": system_prompt,
                "timestamp": dt.datetime.utcnow().isoformat(),
            },
        )

    @staticmethod
    def _extract_flags(handler: CopilotResponseHandler) -> list[str]:
        """Extract flags from the response handler.

        Args:
            handler: The response handler containing generated responses.

        Returns:
            A list of flags indicating special conditions or features.
        """
        flags = {r.response_category.value for r in handler.generated_responses}
        return sorted(flags)

    @staticmethod
    def _full_text(handler: CopilotResponseHandler) -> str:
        """Extract full text from the response handler.

        Args:
            handler: The response handler containing generated responses.

        Returns:
            The concatenated content of all generated responses.
        """
        return "".join(
            response.content
            for response in handler.generated_responses
            if getattr(response, "content", None)
        )

    def log_copilot_from_handler(
        self,
        *,
        handler: CopilotResponseHandler,
        used_documents: list[Document],
        latency_ms: int,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        system_prompt: str,
    ) -> None:
        """Log a copilot message from the response handler.

        Args:
            handler: The response handler containing generated responses.
            used_documents: List of documents used as supporting evidence.
            latency_ms: End-to-end Copilot latency to produce this response.
            model: The model used to generate the response.
            prompt_tokens: Number of input tokens used.
            completion_tokens: Number of output tokens generated.
            total_tokens: Total number of tokens used (input + output).
            system_prompt: The system prompt used.
        """
        structlogger.debug("builder.telemetry.log_copilot_from_handler")
        text = self._full_text(handler)
        self.log_copilot_turn(
            text=text,
            source_urls=[d.url for d in used_documents if d.url],
            flags=self._extract_flags(handler),
            latency_ms=latency_ms,
            model=model,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            system_prompt=system_prompt,
        )
