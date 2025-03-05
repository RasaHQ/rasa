import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

from werkzeug.local import LocalProxy


# Per voice session data
# This is similar to how flask makes the "request" object available as a global variable
# It's a "global" variable that is local to an async task (i.e. websocket session)
@dataclass
class CallState:
    is_user_speaking: bool = False
    is_bot_speaking: bool = False
    silence_timeout_watcher: Optional[asyncio.Task] = None
    silence_timeout: Optional[float] = None
    latest_bot_audio_id: Optional[str] = None
    should_hangup: bool = False
    connection_failed: bool = False

    # Genesys requires the server and client each maintain a
    # monotonically increasing message sequence number.
    client_sequence_number: int = 0
    server_sequence_number: int = 0
    audio_buffer: bytearray = field(default_factory=bytearray)

    # Audiocodes requires a stream ID at start and end of stream
    stream_id: int = 0


_call_state: ContextVar[CallState] = ContextVar("call_state")
call_state = LocalProxy(_call_state)
