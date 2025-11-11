import asyncio
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, cast

from werkzeug.local import LocalProxy

from rasa.shared.core.flows.steps.collect import DTMFConfig


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

    # Latency tracking - start times only
    user_speech_start_time: Optional[float] = None
    rasa_processing_start_time: Optional[float] = None
    tts_start_time: Optional[float] = None

    # Calculated latencies (used by channels like browser_audio)
    asr_latency_ms: Optional[float] = None
    rasa_processing_latency_ms: Optional[float] = None
    tts_first_byte_latency_ms: Optional[float] = None
    tts_complete_latency_ms: Optional[float] = None

    # DTMF State
    is_collecting_dtmf: bool = False
    dtmf_config: Optional[DTMFConfig] = None
    dtmf_buffer: str = ""

    # Generic field for channel-specific state data
    channel_data: Dict[str, Any] = field(default_factory=dict)


_call_state: ContextVar[CallState] = ContextVar("call_state")
call_state: CallState = cast(CallState, LocalProxy(_call_state))
