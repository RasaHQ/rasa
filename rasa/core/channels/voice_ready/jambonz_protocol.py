import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Text

import structlog
from sanic import Websocket  # type: ignore[attr-defined]

from rasa.core.channels.channel import UserMessage
from rasa.core.channels.voice_ready.utils import CallParameters

structlogger = structlog.get_logger()
CHANNEL_NAME = "jambonz"


@dataclass
class NewSessionMessage:
    """Message indicating a new session has been started."""

    call_sid: str
    message_id: str
    call_params: CallParameters

    @staticmethod
    def from_message(message: Dict[str, Any]) -> "NewSessionMessage":
        structlogger.debug("jambonz.websocket.message.new_session", message=message)
        call_params = CallParameters(
            call_id=message.get("call_sid"),
            user_phone=message.get("data", {}).get("from"),
            bot_phone=message.get("data", {}).get("to"),
        )
        return NewSessionMessage(
            message.get("call_sid"),
            message.get("msgid"),
            call_params,
        )


@dataclass
class Transcript:
    """Transcript of a spoken utterance."""

    text: str
    confidence: float


@dataclass
class TranscriptResult:
    """Result of an ASR call with potential transcripts."""

    call_sid: str
    message_id: str
    is_final: bool
    transcripts: List[Transcript] = field(default_factory=list)

    @staticmethod
    def from_speech_result(message: Dict[str, Any]) -> "TranscriptResult":
        return TranscriptResult(
            message.get("call_sid"),
            message.get("msgid"),
            message.get("data", {}).get("speech", {}).get("is_final", True),
            transcripts=[
                Transcript(t.get("transcript", ""), t.get("confidence", 1.0))
                for t in message.get("data", {})
                .get("speech", {})
                .get("alternatives", [])
            ],
        )

    @staticmethod
    def from_dtmf_result(message: Dict[str, Any]) -> "TranscriptResult":
        """Create a transcript result from a DTMF result.

        We use the dtmf as the text with confidence 1.0
        """
        return TranscriptResult(
            message.get("call_sid"),
            message.get("msgid"),
            is_final=True,
            transcripts=[
                Transcript(str(message.get("data", {}).get("digits", "")), 1.0)
            ],
        )


@dataclass
class CallStatusChanged:
    """Message indicating a change in the call status."""

    call_sid: str
    status: str

    @staticmethod
    def from_message(message: Dict[str, Any]) -> "CallStatusChanged":
        structlogger.debug(
            "jambonz.websocket.message.call_status_changed",
            message=message,
        )
        return CallStatusChanged(
            message.get("call_sid"), message.get("data", {}).get("call_status")
        )


@dataclass
class SessionReconnect:
    """Message indicating a session has reconnected."""

    call_sid: str

    @staticmethod
    def from_message(message: Dict[str, Any]) -> "SessionReconnect":
        return SessionReconnect(message.get("call_sid"))


@dataclass
class VerbStatusChanged:
    """Message indicating a change in the status of a verb."""

    call_sid: str
    event: str
    id: str
    name: str

    @staticmethod
    def from_message(message: Dict[str, Any]) -> "VerbStatusChanged":
        return VerbStatusChanged(
            message.get("call_sid"),
            message.get("data", {}).get("event"),
            message.get("data", {}).get("id"),
            message.get("data", {}).get("name"),
        )


@dataclass
class GatherTimeout:
    """Message indicating a gather timeout."""

    call_sid: str

    @staticmethod
    def from_message(message: Dict[str, Any]) -> "GatherTimeout":
        return GatherTimeout(message.get("call_sid"))


async def websocket_message_handler(
    message_dump: str,
    on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ws: Websocket,
) -> None:
    """Handle incoming messages from the websocket."""
    message = json.loads(message_dump)

    # parse and handle the different message types
    if message.get("type") == "session:new":
        new_session = NewSessionMessage.from_message(message)
        await handle_new_session(new_session, on_new_message, ws)
    elif message.get("type") == "session:reconnect":
        session_reconnect = SessionReconnect.from_message(message)
        await handle_session_reconnect(session_reconnect)
    elif message.get("type") == "call:status":
        call_status = CallStatusChanged.from_message(message)
        await handle_call_status(call_status, on_new_message, ws)
    elif message.get("type") == "verb:hook" and message.get("hook") == "/gather":
        hook_trigger_reason = message.get("data", {}).get("reason")

        if hook_trigger_reason == "speechDetected":
            transcript = TranscriptResult.from_speech_result(message)
            await handle_gather_completed(transcript, on_new_message, ws)
        elif hook_trigger_reason == "timeout":
            gather_timeout = GatherTimeout.from_message(message)
            await handle_gather_timeout(gather_timeout, ws)
        elif hook_trigger_reason == "dtmfDetected":
            # for now, let's handle it as normal user input with a
            # confidence of 1.0
            transcript = TranscriptResult.from_dtmf_result(message)
            await handle_gather_completed(transcript, on_new_message, ws)
        else:
            structlogger.debug(
                "jambonz.websocket.message.verb_hook",
                call_sid=message.get("call_sid"),
                reason=hook_trigger_reason,
                message=message,
            )
    elif message.get("type") == "verb:status":
        verb_status = VerbStatusChanged.from_message(message)
        await handle_verb_status(verb_status)
    elif message.get("type") == "jambonz:error":
        # jambonz ran into a fatal error handling the call. the call will be
        # terminated.
        structlogger.error("jambonz.websocket.message.error", message=message)
    else:
        structlogger.warning("jambonz.websocket.message.unknown_type", message=message)


async def handle_new_session(
    message: NewSessionMessage,
    on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ws: Websocket,
) -> None:
    """Handle new session message."""
    from rasa.core.channels.voice_ready.jambonz import JambonzWebsocketOutput

    structlogger.debug("jambonz.websocket.message.new_call", call_sid=message.call_sid)
    output_channel = JambonzWebsocketOutput(ws, message.call_sid)
    user_msg = UserMessage(
        text="/session_start",
        output_channel=output_channel,
        sender_id=message.call_sid,
        metadata=asdict(message.call_params),
        input_channel=CHANNEL_NAME,
    )
    await send_config_ack(message.message_id, ws)
    await on_new_message(user_msg)
    await send_gather_input(ws)


async def handle_gather_completed(
    transcript_result: TranscriptResult,
    on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ws: Websocket,
) -> None:
    """Handle changes to commands we have send to jambonz.

    This includes results of gather calles with their transcription.
    """
    from rasa.core.channels.voice_ready.jambonz import JambonzWebsocketOutput

    if not transcript_result.is_final:
        # in case of a non final transcript, we are going to wait for the final
        # one and ignore the partial one
        structlogger.debug(
            "jambonz.websocket.message.transcript_partial",
            call_sid=transcript_result.call_sid,
            number_of_transcripts=len(transcript_result.transcripts),
        )
        return

    if transcript_result.transcripts:
        most_likely_transcript = transcript_result.transcripts[0]
        output_channel = JambonzWebsocketOutput(ws, transcript_result.call_sid)
        user_msg = UserMessage(
            text=most_likely_transcript.text,
            input_channel=CHANNEL_NAME,
            output_channel=output_channel,
            sender_id=transcript_result.call_sid,
            metadata={},
        )
        structlogger.debug(
            "jambonz.websocket.message.transcript",
            call_sid=transcript_result.call_sid,
            transcript=most_likely_transcript.text,
            confidence=most_likely_transcript.confidence,
            number_of_transcripts=len(transcript_result.transcripts),
        )
        await on_new_message(user_msg)
    else:
        structlogger.warning(
            "jambonz.websocket.message.no_transcript",
            call_sid=transcript_result.call_sid,
        )
    await send_gather_input(ws)


async def handle_gather_timeout(gather_timeout: GatherTimeout, ws: Websocket) -> None:
    """Handle gather timeout."""
    structlogger.debug(
        "jambonz.websocket.message.gather_timeout",
        call_sid=gather_timeout.call_sid,
    )
    # TODO: figure out how to handle timeouts
    await send_ws_text_message(ws, "I'm sorry, I didn't catch that.")
    await send_gather_input(ws)


async def handle_call_status(
    call_status: CallStatusChanged,
    on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ws: Websocket,
) -> None:
    """Handle changes in the call status."""
    structlogger.debug(
        "jambonz.websocket.message.call_status_changed",
        call_sid=call_status.call_sid,
        message=call_status.status,
    )

    if call_status.status == "completed":
        structlogger.debug("jambonz.websocket.message.call_completed")
        from rasa.core.channels.voice_ready.jambonz import JambonzWebsocketOutput

        output_channel = JambonzWebsocketOutput(ws, call_status.call_sid)
        user_msg = UserMessage(
            text="/session_end",
            input_channel=CHANNEL_NAME,
            output_channel=output_channel,
            sender_id=call_status.call_sid,
            metadata={},
        )
        await on_new_message(user_msg)


async def handle_session_reconnect(session_reconnect: SessionReconnect) -> None:
    """Handle session reconnect message."""
    # there is nothing we need to do atm when a session reconnects.
    # this happens if jambonz looses the websocket connection and reconnects
    structlogger.debug(
        "jambonz.websocket.message.session_reconnect",
        call_sid=session_reconnect.call_sid,
    )


async def handle_verb_status(verb_status: VerbStatusChanged) -> None:
    """Handle changes in the status of a verb."""
    structlogger.debug(
        "jambonz.websocket.message.verb_status_changed",
        call_sid=verb_status.call_sid,
        event_type=verb_status.event,
        id=verb_status.id,
        name=verb_status.name,
    )


async def send_config_ack(message_id: str, ws: Websocket) -> None:
    """Send an ack message to jambonz including the configuration."""
    await ws.send(
        json.dumps(
            {
                "type": "ack",
                "msgid": message_id,
                "data": [{"config": {"notifyEvents": True}}],
            }
        )
    )


async def send_gather_input(ws: Websocket) -> None:
    """Send a gather input command to jambonz."""
    structlogger.debug("jambonz.websocket.send.gather")
    await ws.send(
        json.dumps(
            {
                "type": "command",
                "command": "redirect",
                "queueCommand": True,
                "data": [
                    {
                        "gather": {
                            "input": ["speech", "digits"],
                            "minDigits": 1,
                            "id": uuid.uuid4().hex,
                            "actionHook": "/gather",
                        }
                    }
                ],
            }
        )
    )


async def send_ws_text_message(ws: Websocket, text: Text) -> None:
    """Send a text message to the websocket using the jambonz interface."""
    await ws.send(
        json.dumps(
            {
                "type": "command",
                "command": "redirect",
                "queueCommand": True,
                "data": [
                    {
                        "say": {
                            # id can be used for status notifications
                            "id": uuid.uuid4().hex,
                            "text": text,
                        }
                    }
                ],
            }
        )
    )


async def send_ws_hangup_message(hangup_delay_seconds: int, ws: Websocket) -> None:
    """Send a hangup message to the websocket using the jambonz interface."""
    structlogger.debug("jambonz.websocket.send.hangup")
    await ws.send(
        json.dumps(
            {
                "type": "command",
                "command": "redirect",
                "queueCommand": True,
                "data": [
                    {"pause": {"length": hangup_delay_seconds}},
                    {
                        "hangup": {},
                    },
                ],
            }
        )
    )
