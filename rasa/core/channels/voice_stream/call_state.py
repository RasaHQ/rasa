import asyncio
from contextvars import ContextVar
from werkzeug.local import LocalProxy
from dataclasses import dataclass
from typing import Optional


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


_call_state: ContextVar[CallState] = ContextVar("call_state")
call_state = LocalProxy(_call_state)
