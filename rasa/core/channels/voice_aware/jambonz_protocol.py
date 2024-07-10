from dataclasses import dataclass, field
import json
import uuid
from typing import Any, Awaitable, Callable, List, Text

import structlog
from rasa.core.channels.channel import UserMessage
from sanic import Websocket


structlogger = structlog.get_logger()


@dataclass
class NewSessionMessage:
    call_sid: str
    message_id: str


@dataclass
class Transcript:
    text: str
    confidence: float


@dataclass
class TranscriptResult:
    call_sid: str
    message_id: str
    transcripts: List[Transcript] = field(default_factory=list)


@dataclass
class CallStatusChanged:
    call_sid: str
    status: str


async def websocket_message_handler(
    message_dump: str,
    on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ws: Websocket,
) -> None:
    """Handle incoming messages from the websocket."""
    message = json.loads(message_dump)

    if message.get("type") == "session:new":
        new_session = NewSessionMessage(message.get("call_sid"), message.get("msgid"))
        await handle_new_session(new_session, on_new_message, ws)
    elif message.get("type") == "call:status":
        call_status = CallStatusChanged(
            message.get("call_sid"), message.get("data", {}).get("call_status")
        )
        await handle_call_status(call_status)
    elif message.get("type") == "verb:hook":
        if message.get("hook") == "/transcript":
            transcript = TranscriptResult(
                message.get("call_sid"),
                message.get("msgid"),
                transcripts=[
                    Transcript(t.get("transcript", ""), t.get("confidence", 1.0))
                    for t in message.get("data", {})
                    .get("speech", {})
                    .get("alternatives", [])
                ],
            )
            await handle_verb_hook(transcript, on_new_message, ws)
    else:
        structlogger.warning("jambonz.websocket.message.unknown_type", message=message)


async def handle_new_session(
    message: NewSessionMessage,
    on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ws: Websocket,
) -> None:
    """Handle new session message."""
    from rasa.core.channels.voice_aware.jambonz import JambonzWebsocketOutput

    structlogger.debug("jambonz.websocket.message.new_call", call_sid=message.call_sid)
    output_channel = JambonzWebsocketOutput(ws, message.call_sid)
    user_msg = UserMessage(
        text="/session_start",
        output_channel=output_channel,
        sender_id=message.call_id,
        metadata={},
    )
    await send_config_ack(message.message_id, ws)
    await on_new_message(user_msg)
    await send_gather_input(ws)


async def handle_verb_hook(
    transcript_result: TranscriptResult,
    on_new_message: Callable[[UserMessage], Awaitable[Any]],
    ws: Websocket,
) -> None:
    """Handle changes to commands we have send to jambonz.

    This includes results of gather calles with their transcription."""
    from rasa.core.channels.voice_aware.jambonz import JambonzWebsocketOutput

    if transcript_result.transcripts:
        most_likely_transcript = transcript_result.transcripts[0]
        output_channel = JambonzWebsocketOutput(ws, transcript_result.call_sid)
        user_msg = UserMessage(
            text=most_likely_transcript.text,
            output_channel=output_channel,
            sender_id=transcript_result.call_sid,
            metadata={},
        )
        structlogger.debug(
            "jambonz.websocket.message.transcript",
            call_sid=transcript_result.call_id,
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


async def handle_call_status(call_status: CallStatusChanged) -> None:
    structlogger.debug(
        "jambonz.websocket.message.call_status_changed",
        call_sid=call_status.call_sid.get("call_sid"),
        message=call_status.status,
    )


async def send_config_ack(message_id: str, ws: Websocket) -> None:
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
    await ws.send(
        json.dumps(
            {
                "type": "command",
                "command": "redirect",
                "queueCommand": True,
                "data": [
                    {
                        "gather": {
                            "input": ["speech"],
                            "id": uuid.uuid4().hex,
                            "actionHook": "/transcript",
                        }
                    }
                ],
            }
        )
    )


async def send_ws_text_message(ws: Websocket, text: Text) -> None:
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
