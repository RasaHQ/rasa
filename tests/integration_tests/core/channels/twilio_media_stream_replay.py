"""
Test utility for replaying Twilio Media Stream WebSocket traffic.

Replays captured Twilio Media Stream WebSocket traffic for testing Rasa.
"""

import argparse
import asyncio
import base64
import json
import logging

from aiohttp import ClientSession, WSMsgType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TWILIO_LOG_TYPE = "twilio_media_stream"


def _create_start_message(
    stream_sid: str,
    call_id: str,
    user_phone: str,
    bot_phone: str,
    direction: str = "inbound",
) -> dict:
    """Build Twilio 'start' event JSON (same shape as channel expects)."""
    return {
        "event": "start",
        "sequenceNumber": "1",
        "start": {
            "accountSid": "ACbc2d4fd426ce33de19d54bdcd6e41186",
            "streamSid": stream_sid,
            "callSid": call_id,
            "tracks": ["inbound"],
            "mediaFormat": {
                "encoding": "audio/x-mulaw",
                "sampleRate": 8000,
                "channels": 1,
            },
            "customParameters": {
                "direction": direction,
                "call_id": call_id,
                "user_phone": user_phone,
                "bot_phone": bot_phone,
            },
        },
        "streamSid": stream_sid,
    }


def _create_media_messages_from_audio_bytes(
    audio_bytes: bytes,
    stream_sid: str,
    chunk_size: int = 1024,
    start_sequence: int = 2,
) -> list:
    """Build list of Twilio 'media' event dicts from raw audio bytes (e.g. μ-law).

    start_sequence: first sequence number for media messages (start is 1, so default 2).
    """
    messages = []
    offset = 0
    i = 0
    while offset < len(audio_bytes):
        chunk = audio_bytes[offset : offset + chunk_size]
        payload_b64 = base64.b64encode(chunk).decode("utf-8")
        messages.append(
            {
                "event": "media",
                "sequenceNumber": str(start_sequence + i),
                "media": {
                    "track": "inbound",
                    "chunk": str(i),
                    "timestamp": str(offset // 8),
                    "payload": payload_b64,
                },
                "streamSid": stream_sid,
            }
        )
        i += 1
        offset += chunk_size
    return messages


def _create_stop_message(stream_sid: str, sequence_number: int) -> dict:
    """Build Twilio 'stop' event JSON with the next sequential sequence number."""
    return {
        "event": "stop",
        "sequenceNumber": str(sequence_number),
        "streamSid": stream_sid,
    }


class TwilioMediaStreamReplay:
    """Replay a Twilio Media Stream session from a traffic log."""

    def __init__(
        self, rasa_url: str, log_file: str, timeout: int = 5, delay: float = 1.0
    ):
        self.rasa_url = rasa_url
        self.log_file = log_file
        self.timeout = timeout
        self.delay = delay
        self.connection_state = {
            "connected": False,
            "start_sent": False,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": [],
        }

    def _get_incoming(self, traffic_log: list) -> list:
        """Filter log for incoming Twilio WebSocket messages."""
        return [
            entry
            for entry in traffic_log
            if entry.get("direction") == "incoming"
            and entry.get("type") in (TWILIO_LOG_TYPE, "websocket")
        ]

    def _get_outgoing(self, traffic_log: list) -> list:
        """Filter log for outgoing Twilio WebSocket messages."""
        return [
            entry
            for entry in traffic_log
            if entry.get("direction") == "outgoing"
            and entry.get("type") in (TWILIO_LOG_TYPE, "websocket")
        ]

    def _validate_message_flow(self, incoming: list) -> dict:
        """Validate that the message flow follows Twilio expectations."""
        warnings = []
        if not incoming:
            return {"valid": True, "warnings": []}
        first = incoming[0].get("data", {})
        event = first.get("event")
        if event != "start":
            warnings.append(f"First message must be event 'start', got '{event}'")
        if event == "start":
            if not first.get("streamSid"):
                warnings.append("'start' message must include 'streamSid'")
            start = first.get("start", {})
            params = start.get("customParameters", {})
            for key in ("call_id", "user_phone", "bot_phone"):
                if key not in params:
                    warnings.append(
                        f"'start.start.customParameters' should contain '{key}'"
                    )
        return {"valid": len(warnings) == 0, "warnings": warnings}

    def _should_wait_for_response(self, event: str) -> bool:
        """Whether to wait for a response after sending this event."""
        return event not in ("media",)

    async def replay_twilio_session(self) -> None:
        """Replay a complete Twilio Media Stream session from the traffic log."""
        logger.info("Loading traffic from %s", self.log_file)
        try:
            with open(self.log_file, "r") as f:
                traffic_log = json.load(f)
        except FileNotFoundError:
            logger.error("File not found: %s", self.log_file)
            return
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in log file: %s", e)
            return

        incoming = self._get_incoming(traffic_log)
        outgoing = self._get_outgoing(traffic_log)
        logger.debug(
            "Found %s incoming and %s outgoing Twilio WebSocket messages",
            len(incoming),
            len(outgoing),
        )

        if not incoming:
            logger.debug("No incoming Twilio messages found in log file")
            return

        validation_result = self._validate_message_flow(incoming)
        if not validation_result["valid"]:
            logger.warning(
                "Message flow validation warnings: %s",
                validation_result["warnings"],
            )
            for warning in validation_result["warnings"]:
                logger.warning("  - %s", warning)

        base_ws = self.rasa_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        ws_url = f"{base_ws}/webhooks/twilio_media_streams/websocket"
        logger.info("Connecting to Rasa at %s", ws_url)

        async with ClientSession() as session:
            try:
                async with session.ws_connect(ws_url) as ws:
                    self.connection_state["connected"] = True
                    logger.info("✓ WebSocket connected")

                    for i, entry in enumerate(incoming, 1):
                        data = entry["data"]
                        event = data.get("event", "unknown")
                        logger.debug(
                            "[%s/%s] Sending event: %s", i, len(incoming), event
                        )
                        logger.debug("Data: %s", json.dumps(data, indent=2))

                        if ws.closed:
                            logger.debug(
                                "Connection closed before sending message %s (%s)",
                                i,
                                event,
                            )
                            self.connection_state["errors"].append(
                                {
                                    "message_index": i,
                                    "event": event,
                                    "error": "Connection closed before sending",
                                }
                            )
                            break

                        try:
                            await ws.send_json(data)
                            self.connection_state["messages_sent"] += 1
                            if event == "start":
                                self.connection_state["start_sent"] = True
                        except Exception as send_error:
                            error_str = str(send_error)
                            error_type = type(send_error).__name__
                            is_graceful = (
                                "closing transport" in error_str.lower()
                                or "connection closed" in error_str.lower()
                                or error_type
                                in (
                                    "ConnectionResetError",
                                    "ClientConnectionResetError",
                                )
                            )
                            self.connection_state["errors"].append(
                                {
                                    "message_index": i,
                                    "event": event,
                                    "error": error_str,
                                    "graceful": is_graceful,
                                }
                            )
                            if is_graceful:
                                logger.debug(
                                    "Connection closed by Rasa while sending "
                                    "message %s (%s) - expected when conversation ends",
                                    i,
                                    event,
                                )
                            else:
                                logger.debug(
                                    "Error sending message %s (%s): %s",
                                    i,
                                    event,
                                    send_error,
                                )
                            break

                        if self._should_wait_for_response(event):
                            try:
                                msg = await asyncio.wait_for(
                                    ws.receive(), timeout=self.timeout
                                )
                                self.connection_state["messages_received"] += 1
                                if msg.type == WSMsgType.TEXT:
                                    response = json.loads(msg.data)
                                    resp_event = response.get("event", "unknown")
                                    logger.debug(
                                        "← Response: %s",
                                        json.dumps(response, indent=2),
                                    )
                                    if resp_event == "media":
                                        logger.debug("Received media from Rasa")
                                elif msg.type == WSMsgType.CLOSE:
                                    close_code = (
                                        msg.data if hasattr(msg, "data") else "unknown"
                                    )
                                    # 1000 = Normal Closure (RFC 6455)
                                    is_graceful_close = close_code == 1000
                                    logger.warning(
                                        "WebSocket closed by server (code: %s)",
                                        close_code,
                                    )
                                    self.connection_state["errors"].append(
                                        {
                                            "message_index": i,
                                            "event": event,
                                            "error": (
                                                "Connection closed by server "
                                                "(code: %s)" % close_code
                                            ),
                                            "graceful": is_graceful_close,
                                        }
                                    )
                                    break
                            except asyncio.TimeoutError:
                                logger.debug(
                                    "No response for %s (timeout after %ss)",
                                    event,
                                    self.timeout,
                                )
                        else:
                            logger.debug("Skipping response wait for event '%s'", event)

                        if i < len(incoming):
                            await asyncio.sleep(self.delay)

                    self._print_session_summary()
                    logger.info("✓ Replay complete")

                    if not ws.closed:
                        try:
                            await ws.close()
                            logger.info("✓ Connection closed gracefully")
                        except Exception as close_error:
                            logger.warning("Error closing connection: %s", close_error)

            except Exception as e:
                logger.error("WebSocket error: %s", e, exc_info=True)
                error_str = str(e)
                is_graceful = (
                    "closing transport" in error_str.lower()
                    or "connection closed" in error_str.lower()
                    or type(e).__name__
                    in (
                        "ConnectionResetError",
                        "ClientConnectionResetError",
                    )
                )
                self.connection_state["errors"].append(
                    {
                        "error_type": type(e).__name__,
                        "error_message": error_str,
                        "graceful": is_graceful,
                    }
                )
                self._print_session_summary()

    def _print_session_summary(self) -> None:
        """Print a summary of the replay session."""
        logger.info("\n" + "=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.debug(
            "Messages sent:      %s",
            self.connection_state["messages_sent"],
        )
        logger.debug(
            "Messages received: %s",
            self.connection_state["messages_received"],
        )
        logger.info(
            "Start sent:         %s",
            self.connection_state["start_sent"],
        )
        errors = self.connection_state["errors"]
        if errors:
            graceful = [e for e in errors if e.get("graceful", False)]
            actual = [e for e in errors if not e.get("graceful", False)]
            if graceful:
                logger.info("\nGraceful connection closures: %s", len(graceful))
            if actual:
                logger.warning("\nErrors encountered: %s", len(actual))
            elif not graceful:
                logger.info("\nNo errors encountered")
        else:
            logger.info("\nNo errors encountered")
        logger.info("=" * 60 + "\n")

    async def analyze_traffic(self) -> None:
        """Analyze captured traffic and show statistics."""
        logger.info("Analyzing traffic from %s", self.log_file)
        try:
            with open(self.log_file, "r") as f:
                traffic_log = json.load(f)
        except Exception as e:
            logger.error("Error loading file: %s", e)
            return

        incoming = self._get_incoming(traffic_log)
        outgoing = self._get_outgoing(traffic_log)
        total = len(traffic_log)

        print("\n" + "=" * 60)
        print("TWILIO MEDIA STREAM TRAFFIC ANALYSIS")
        print("=" * 60)
        print(f"Total messages:     {total}")
        print(f"Incoming:           {len(incoming)}")
        print(f"Outgoing:           {len(outgoing)}")
        print("=" * 60)

        event_counts: dict = {}
        for entry in incoming:
            ev = entry.get("data", {}).get("event", "unknown")
            event_counts[ev] = event_counts.get(ev, 0) + 1
        print("\nINCOMING EVENTS:")
        for ev, count in sorted(event_counts.items()):
            print(f"  {ev}: {count}")

        event_counts_out: dict = {}
        for entry in outgoing:
            ev = entry.get("data", {}).get("event", "unknown")
            event_counts_out[ev] = event_counts_out.get(ev, 0) + 1
        print("\nOUTGOING EVENTS:")
        for ev, count in sorted(event_counts_out.items()):
            print(f"  {ev}: {count}")

        print("\nFIRST 3 INCOMING MESSAGES:")
        for idx, entry in enumerate(incoming[:3], 1):
            ev = entry.get("data", {}).get("event")
            print(f"\n{idx}. event={ev}")
            print("   " + json.dumps(entry.get("data", {}), indent=2))

        print("\n" + "=" * 60 + "\n")


def generate_sample_log(
    wav_path: str,
    output_path: str,
    stream_sid: str = "MZcdce5426d49ccf48c7b0d0ab86a63d52",
    call_id: str = "CAa874cb4d1ac15290b51b28c91d467812",
    user_phone: str = "+49176124567",
    bot_phone: str = "+49123456789",
    direction: str = "inbound",
    chunk_size: int = 1024,
) -> None:
    """Generate a sample Twilio traffic log from a WAV file (8 kHz μ-law)."""
    from rasa.core.channels.voice_stream.util import read_wav_to_rasa_audio_bytes

    rasa_audio = read_wav_to_rasa_audio_bytes(wav_path)
    if rasa_audio is None:
        raise ValueError(f"Could not read WAV as Rasa audio bytes: {wav_path}")
    audio_bytes = bytes(rasa_audio.data)

    start_msg = _create_start_message(
        stream_sid, call_id, user_phone, bot_phone, direction
    )
    media_msgs = _create_media_messages_from_audio_bytes(
        audio_bytes, stream_sid, chunk_size=chunk_size, start_sequence=2
    )
    stop_msg = _create_stop_message(stream_sid, sequence_number=2 + len(media_msgs))

    log_entries = []
    log_entries.append(
        {"direction": "incoming", "type": TWILIO_LOG_TYPE, "data": start_msg}
    )
    for msg in media_msgs:
        log_entries.append(
            {"direction": "incoming", "type": TWILIO_LOG_TYPE, "data": msg}
        )
    log_entries.append(
        {"direction": "incoming", "type": TWILIO_LOG_TYPE, "data": stop_msg}
    )

    with open(output_path, "w") as f:
        json.dump(log_entries, f, indent=2)
    logger.info("Sample Twilio traffic log written.")
    logger.debug(
        "Wrote sample Twilio traffic log with %s messages to %s",
        len(log_entries),
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Twilio Media Stream traffic replay and analysis"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=None,
        help="Path to twilio_traffic.json (required for replay/analyze)",
    )
    parser.add_argument(
        "--rasa-url",
        default="http://localhost:5005",
        help="Rasa server URL (default: http://localhost:5005)",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "analyze", "generate"],
        default="replay",
        help="Mode: replay, analyze, or generate sample log from WAV",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds for waiting for responses (default: 5)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between messages (default: 1.0)",
    )
    parser.add_argument(
        "--wav",
        help="Path to WAV file (8 kHz μ-law) for generate mode",
    )
    parser.add_argument(
        "--output",
        default="twilio_traffic.json",
        help="Output path for generate mode (default: twilio_traffic.json)",
    )
    parser.add_argument(
        "--stream-sid",
        default="MZcdce5426d49ccf48c7b0d0ab86a63d52",
        help="Stream SID for generate mode",
    )
    parser.add_argument(
        "--call-id",
        default="CAa874cb4d1ac15290b51b28c91d467812",
        help="Call SID for generate mode",
    )
    parser.add_argument(
        "--user-phone",
        default="+49176124567",
        help="User phone for generate mode",
    )
    parser.add_argument(
        "--bot-phone",
        default="+49123456789",
        help="Bot phone for generate mode",
    )
    args = parser.parse_args()

    if args.mode == "generate":
        if not args.wav:
            parser.error("--wav is required for generate mode")
        generate_sample_log(
            args.wav,
            args.output,
            stream_sid=args.stream_sid,
            call_id=args.call_id,
            user_phone=args.user_phone,
            bot_phone=args.bot_phone,
        )
        return

    if not args.log_file:
        parser.error("log_file is required for replay and analyze modes")
    replay = TwilioMediaStreamReplay(
        args.rasa_url, args.log_file, timeout=args.timeout, delay=args.delay
    )
    if args.mode == "replay":
        asyncio.run(replay.replay_twilio_session())
    else:
        asyncio.run(replay.analyze_traffic())


if __name__ == "__main__":
    main()
