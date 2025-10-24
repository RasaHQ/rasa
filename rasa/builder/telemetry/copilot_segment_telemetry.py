import datetime as dt
import os
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Optional,
    Sequence,
)

if TYPE_CHECKING:
    pass

import structlog

from rasa import telemetry
from rasa.builder.copilot.constants import COPILOT_SEGMENT_WRITE_KEY_ENV_VAR
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.models import (
    EventContent,
)
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


class CopilotSegmentTelemetry:
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
        cached_prompt_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        system_message: Optional[dict[str, Any]] = None,
        chat_history: Optional[list[dict[str, Any]]] = None,
        last_user_message: Optional[str] = None,
        tracker_event_attachments: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Track a copilot message in the conversation.

        Args:
            text: The text of the copilot message.
            source_urls: URLs of the sources used to generate the response.
            flags: Flags indicating special conditions or features.
            latency_ms: End-to-end Copilot latency to produce this response.
            model: The model used to generate the response.
            input_tokens: Number of input tokens used (optional).
            cached_prompt_tokens: Number of cached prompt tokens.
            output_tokens: Number of output tokens generated (optional).
            total_tokens: Total number of tokens used (input + output) (optional).
            system_message: The system message used (optional).
            chat_history: The chat history messages used (optional).
            last_user_message: The last user message used (optional).
            tracker_event_attachments: The tracker event attachments used (optional).
        """
        structlogger.debug("builder.telemetry.log_copilot_turn", text=text)

        # FIXME: Temporarily remove the system_message from telemetry payload.
        # Reason: It often exceeds Segment payload size limits, causing the request
        # to be rejected and the event to be absent in Segment. Instead, temporarily
        # log the system_message so it's visible in Grafana.
        telemetry_data = {
            "project_id": self._project_id,
            "message_id": uuid.uuid4().hex,
            "text": text,
            "prompt_version": self._prompt_version,
            "source_urls": list(source_urls),
            "flags": list(flags),
            "latency_ms": latency_ms,
            "model": model,
            "input_tokens": input_tokens,
            "cached_prompt_tokens": cached_prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "chat_history": chat_history,
            "last_user_message": last_user_message,
            "tracker_event_attachments": tracker_event_attachments,
            "timestamp": dt.datetime.utcnow().isoformat(),
        }

        # Log all telemetry data plus system_message for debugging
        log_data = telemetry_data.copy()
        if system_message:
            log_data["system_message"] = system_message

        structlogger.info("builder.telemetry.copilot_turn", **log_data)

        _track(
            COPILOT_BOT_MESSAGE_EVENT,
            self._user_id,
            telemetry_data,
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
        cached_prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        system_message: dict[str, Any],
        chat_history: list[dict[str, Any]],
        last_user_message: Optional[str],
        tracker_event_attachments: list[EventContent],
    ) -> None:
        """Log a copilot message from the response handler.

        Args:
            handler: The response handler containing generated responses.
            used_documents: List of documents used as supporting evidence.
            latency_ms: End-to-end Copilot latency to produce this response.
            model: The model used to generate the response.
            prompt_tokens: Number of input tokens used.
            cached_prompt_tokens: Number of cached prompt tokens.
            completion_tokens: Number of output tokens generated.
            total_tokens: Total number of tokens used (input + output).
            system_message: The system message used.
            chat_history: The chat history messages used.
            last_user_message: The last user message used.
            tracker_event_attachments: List of tracker event attachments.
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
            cached_prompt_tokens=cached_prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=total_tokens,
            system_message=system_message,
            chat_history=chat_history,
            last_user_message=last_user_message,
            tracker_event_attachments=[
                attachment.model_dump() for attachment in tracker_event_attachments
            ],
        )
