"""
Test utility for replaying Jambonz Stream WebSocket traffic.

Replays captured Jambonz Stream WebSocket traffic for testing Rasa.
Jambonz Stream protocol: first message is JSON metadata (callSid, from, to),
then binary L16 PCM audio or JSON (e.g. dtmf, mark) messages.
"""

import argparse
import asyncio
import base64
import json
import logging

from aiohttp import ClientSession, WSMsgType

from tests.integration_tests.core.channels.utils.replay_common import (
    build_websocket_url,
    is_graceful_connection_error,
    load_traffic_log,
    print_session_summary,
    record_connection_closed_before_send,
    record_outer_error,
    record_send_error,
    record_server_close,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

JAMBONZ_LOG_TYPE = "jambonz_stream"
JAMBONZ_WEBSOCKET_PATH = "webhooks/jambonz_stream/websocket"
JAMBONZ_SUBPROTOCOL = "audio.jambonz.org"


def _is_metadata_message(data: dict) -> bool:
    """Check if this is the initial Jambonz metadata (call params) message."""
    return "callSid" in data and "from" in data


class JambonzStreamReplay:
    """Replay a Jambonz Stream session from a traffic log."""

    def __init__(
        self, rasa_url: str, log_file: str, timeout: int = 5, delay: float = 1.0
    ):
        self.rasa_url = rasa_url
        self.log_file = log_file
        self.timeout = timeout
        self.delay = delay
        self.connection_state = {
            "connected": False,
            "metadata_sent": False,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": [],
        }

    def _get_incoming(self, traffic_log: list) -> list:
        """Filter log for incoming Jambonz WebSocket messages."""
        return [
            entry
            for entry in traffic_log
            if entry.get("direction") == "incoming"
            and entry.get("type") in (JAMBONZ_LOG_TYPE, "websocket")
        ]

    def _validate_message_flow(self, incoming: list) -> dict:
        """Validate that the message flow follows Jambonz expectations."""
        warnings = []
        if not incoming:
            return {"valid": True, "warnings": []}
        first = incoming[0].get("data", {})
        if isinstance(first, dict) and not _is_metadata_message(first):
            warnings.append(
                "First message must be JSON metadata with callSid, from, to"
            )
        return {"valid": len(warnings) == 0, "warnings": warnings}

    async def replay_jambonz_session(self) -> None:
        """Replay a complete Jambonz Stream session from the traffic log."""
        logger.info("Loading traffic from %s", self.log_file)
        traffic_log = load_traffic_log(self.log_file)
        if traffic_log is None:
            return

        incoming = self._get_incoming(traffic_log)
        logger.debug(
            "Found %s incoming Jambonz WebSocket messages",
            len(incoming),
        )

        if not incoming:
            logger.debug("No incoming Jambonz messages found in log file")
            return

        validation_result = self._validate_message_flow(incoming)
        if not validation_result["valid"]:
            logger.warning(
                "Message flow validation warnings: %s",
                validation_result["warnings"],
            )
            for warning in validation_result["warnings"]:
                logger.warning("  - %s", warning)

        ws_url = build_websocket_url(self.rasa_url, JAMBONZ_WEBSOCKET_PATH)
        logger.info(
            "Connecting to Rasa at %s (subprotocol: %s)",
            ws_url,
            JAMBONZ_SUBPROTOCOL,
        )

        async with ClientSession() as session:
            try:
                async with session.ws_connect(
                    ws_url, protocols=[JAMBONZ_SUBPROTOCOL]
                ) as ws:
                    self.connection_state["connected"] = True
                    logger.info("✓ WebSocket connected")

                    for i, entry in enumerate(incoming, 1):
                        data = entry.get("data")
                        is_binary = entry.get("binary", False)
                        payload_b64 = entry.get("payload_base64")

                        message_kind = "binary" if is_binary else "json"
                        if ws.closed:
                            logger.debug(
                                "Connection closed before sending message %s",
                                i,
                            )
                            record_connection_closed_before_send(
                                self.connection_state["errors"], i, message_kind
                            )
                            break

                        try:
                            if is_binary and payload_b64:
                                raw = base64.b64decode(payload_b64)
                                await ws.send_bytes(raw)
                            else:
                                payload = (
                                    data if isinstance(data, str) else json.dumps(data)
                                )
                                await ws.send_str(payload)
                            self.connection_state["messages_sent"] += 1
                            if (
                                i == 1
                                and isinstance(data, dict)
                                and _is_metadata_message(data)
                            ):
                                self.connection_state["metadata_sent"] = True
                        except Exception as send_error:
                            if is_graceful_connection_error(send_error):
                                logger.debug(
                                    "Connection closed by Rasa while sending "
                                    "message %s - expected when conversation ends",
                                    i,
                                )
                            else:
                                logger.debug(
                                    "Error sending message %s: %s",
                                    i,
                                    send_error,
                                )
                            record_send_error(
                                self.connection_state["errors"],
                                i,
                                message_kind,
                                send_error,
                            )
                            break

                        try:
                            msg = await asyncio.wait_for(
                                ws.receive(), timeout=self.timeout
                            )
                            self.connection_state["messages_received"] += 1
                            if msg.type == WSMsgType.TEXT:
                                logger.debug(
                                    "← Text response: %s",
                                    msg.data[:200] if msg.data else "",
                                )
                            elif msg.type == WSMsgType.BINARY:
                                logger.debug(
                                    "← Binary response: %s bytes",
                                    len(msg.data),
                                )
                            elif msg.type == WSMsgType.CLOSE:
                                close_code = (
                                    msg.data if hasattr(msg, "data") else "unknown"
                                )
                                logger.warning(
                                    "WebSocket closed by server (code: %s)",
                                    close_code,
                                )
                                record_server_close(
                                    self.connection_state["errors"],
                                    i,
                                    message_kind,
                                    close_code,
                                )
                                break
                        except asyncio.TimeoutError:
                            logger.debug(
                                "No response for message %s (timeout after %ss)",
                                i,
                                self.timeout,
                            )

                        if i < len(incoming):
                            await asyncio.sleep(self.delay)

                    print_session_summary(
                        self.connection_state,
                        state_keys=[("metadata_sent", "Metadata sent")],
                    )
                    logger.info("✓ Replay complete")

                    if not ws.closed:
                        try:
                            await ws.close()
                            logger.info("✓ Connection closed gracefully")
                        except Exception as close_error:
                            logger.warning("Error closing connection: %s", close_error)

            except Exception as e:
                logger.error("WebSocket error: %s", e, exc_info=True)
                record_outer_error(self.connection_state["errors"], e)
                print_session_summary(
                    self.connection_state,
                    state_keys=[("metadata_sent", "Metadata sent")],
                )


def generate_sample_log(
    output_path: str,
    call_sid: str = "test-call-jambonz-001",
    from_number: str = "+15551234567",
    to_number: str = "+15559876543",
) -> None:
    """Generate a minimal Jambonz Stream traffic log (metadata only)."""
    metadata_msg = {
        "direction": "incoming",
        "type": JAMBONZ_LOG_TYPE,
        "data": {
            "callSid": call_sid,
            "from": from_number,
            "to": to_number,
        },
    }
    log_entries = [metadata_msg]
    with open(output_path, "w") as f:
        json.dump(log_entries, f, indent=2)
    logger.info("Generated Jambonz traffic log")
    logger.debug(
        "Generated Jambonz traffic log with %s message(s) to %s",
        len(log_entries),
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jambonz Stream traffic replay and generation"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=None,
        help="Path to jambonz_traffic.json (required for replay mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "generate"],
        default="replay",
        help="Mode: replay or generate (default: replay)",
    )
    parser.add_argument(
        "--output",
        default="data/e2e_voice/jambonz_traffic.json",
        help="Output path for generate mode",
    )
    parser.add_argument(
        "--call-sid",
        default="test-call-jambonz-001",
        help="Call SID for generate mode",
    )
    parser.add_argument(
        "--from",
        dest="from_number",
        default="+15551234567",
        help="From number for generate mode",
    )
    parser.add_argument(
        "--to",
        dest="to_number",
        default="+15559876543",
        help="To number for generate mode",
    )
    parser.add_argument(
        "--rasa-url",
        default="http://localhost:5005",
        help="Rasa server URL (default: http://localhost:5005)",
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
    args = parser.parse_args()

    if args.mode == "generate":
        generate_sample_log(
            args.output,
            call_sid=args.call_sid,
            from_number=args.from_number,
            to_number=args.to_number,
        )
        return

    if not args.log_file:
        parser.error("log_file is required for replay mode")
    replay = JambonzStreamReplay(
        args.rasa_url, args.log_file, timeout=args.timeout, delay=args.delay
    )
    asyncio.run(replay.replay_jambonz_session())
