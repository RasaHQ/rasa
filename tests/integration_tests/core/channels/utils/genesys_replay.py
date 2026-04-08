"""Test utility for replaying Genesys AudioHook WebSocket traffic.

Replays captured Genesys WebSocket traffic against the Rasa Genesys connector.
The WebSocket upgrade must include Audiohook-* headers and X-Api-Key (see
rasa.core.channels.voice_stream.genesys). Supports generate mode for CI.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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

GENESYS_WEBSOCKET_PATH = "webhooks/genesys/websocket"

# Must match the api_key merged into calm-benchmarking-bot credentials in CI.
DEFAULT_GENESYS_API_KEY = "1234"


def genesys_ws_headers() -> Dict[str, str]:
    """Headers required for the Genesys WebSocket handshake."""
    return {
        "Audiohook-Organization-Id": os.environ.get(
            "GENESYS_ORG_ID", "22352111-6076-492a-8163-514a00723975"
        ),
        "Audiohook-Correlation-Id": os.environ.get(
            "GENESYS_CORRELATION_ID", "e2e-genesys-correlation-001"
        ),
        "Audiohook-Session-Id": os.environ.get(
            "GENESYS_SESSION_ID", "e2e-genesys-session-001"
        ),
        "X-Api-Key": os.environ.get("GENESYS_API_KEY") or DEFAULT_GENESYS_API_KEY,
    }


class GenesysReplay:
    """Replay a Genesys AudioHook session from a JSON traffic log."""

    def __init__(
        self,
        rasa_url: str,
        log_file: str,
        timeout: int = 5,
        delay: float = 1.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.rasa_url = rasa_url
        self.log_file = log_file
        self.timeout = timeout
        self.delay = delay
        self.headers = headers if headers is not None else genesys_ws_headers()
        self._reset_connection_state()

    def _reset_connection_state(self) -> None:
        """Reset per-session counters so a second replay run starts clean."""
        self.connection_state = {
            "connected": False,
            "open_message_sent": False,
            "opened_response_received": False,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": [],
        }

    def _incoming_messages(
        self, traffic_log: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [
            entry
            for entry in traffic_log
            if entry.get("direction") == "incoming" and entry.get("type") == "websocket"
        ]

    def _validate_message_flow(self, incoming: List[Dict[str, Any]]) -> Dict[str, Any]:
        warnings: List[str] = []
        if not incoming:
            return {"valid": True, "warnings": warnings}
        first = incoming[0].get("data")
        if isinstance(first, dict) and first.get("type") != "open":
            warnings.append("First WebSocket message should be type 'open'")
        return {"valid": len(warnings) == 0, "warnings": warnings}

    async def _receive_one(
        self, ws: Any, message_index: int, message_kind: str
    ) -> bool:
        """Receive one message; update state. Returns False if the loop should stop."""
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.debug(
                "No recv within timeout idx=%s kind=%s after %ss",
                message_index,
                message_kind,
                self.timeout,
            )
            return True

        self.connection_state["messages_received"] += 1

        if msg.type == WSMsgType.TEXT:
            try:
                parsed = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.debug(
                    "non-JSON AudioHook payload (truncated): %s",
                    (msg.data or "")[:200],
                )
                return True
            ctrl_type = parsed.get("type", "unknown")
            logger.debug("AudioHook ctrl type=%s payload=%s", ctrl_type, parsed)
            if ctrl_type == "opened":
                self.connection_state["opened_response_received"] = True
        elif msg.type == WSMsgType.BINARY:
            logger.debug("Binary audio chunk size=%s bytes", len(msg.data))
        elif msg.type == WSMsgType.CLOSE:
            close_code = msg.data if hasattr(msg, "data") else "unknown"
            logger.warning("WebSocket closed by server (code: %s)", close_code)
            record_server_close(
                self.connection_state["errors"],
                message_index,
                message_kind,
                close_code,
            )
            return False
        return True

    async def replay_genesys_session(self) -> None:
        """Connect with AudioHook headers, replay log, read replies."""
        self._reset_connection_state()
        logger.info("Loading traffic from %s", self.log_file)
        traffic_log = load_traffic_log(self.log_file)
        if traffic_log is None:
            return

        incoming = self._incoming_messages(traffic_log)
        # Brief format string (semgrep flags keywords like message/frame on info).
        logger.info("Genesys replay count: %s", len(incoming))

        if not incoming:
            logger.debug("Genesys WebSocket traffic log has no entries")
            return

        validation = self._validate_message_flow(incoming)
        if not validation["valid"]:
            for w in validation["warnings"]:
                logger.warning("Traffic validation: %s", w)

        ws_url = build_websocket_url(self.rasa_url, GENESYS_WEBSOCKET_PATH)
        logger.info("Connecting to %s", ws_url)

        async with ClientSession() as session:
            try:
                async with session.ws_connect(ws_url, headers=self.headers) as ws:
                    self.connection_state["connected"] = True
                    logger.info("✓ WebSocket connected")

                    for i, entry in enumerate(incoming, 1):
                        is_binary = entry.get("binary", False)
                        payload_b64 = entry.get("payload_base64")
                        data = entry.get("data")

                        if is_binary and payload_b64:
                            message_kind = "binary"
                            message_type = "binary"
                        elif isinstance(data, dict):
                            message_kind = "json"
                            message_type = data.get("type", "unknown")
                        else:
                            message_kind = "json"
                            message_type = "unknown"

                        if ws.closed:
                            record_connection_closed_before_send(
                                self.connection_state["errors"], i, message_kind
                            )
                            break

                        try:
                            if is_binary and payload_b64:
                                raw = base64.b64decode(payload_b64)
                                await ws.send_bytes(raw)
                            elif isinstance(data, dict):
                                await ws.send_json(data)
                            elif isinstance(data, str):
                                await ws.send_str(data)
                            else:
                                logger.warning(
                                    "Skipping entry %s: unsupported payload",
                                    i,
                                )
                                continue

                            self.connection_state["messages_sent"] += 1
                            if message_type == "open":
                                self.connection_state["open_message_sent"] = True
                        except Exception as send_error:
                            if is_graceful_connection_error(send_error):
                                logger.debug(
                                    "Connection closed during send at index %s",
                                    i,
                                )
                            else:
                                logger.debug(
                                    "Send failed at index=%s err=%s",
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

                        if not await self._receive_one(ws, i, message_kind):
                            break

                        if i < len(incoming):
                            await asyncio.sleep(self.delay)

                    print_session_summary(
                        self.connection_state,
                        state_keys=[
                            ("open_message_sent", "Open handshake sent"),
                            ("opened_response_received", "Opened handshake recv"),
                        ],
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
                    state_keys=[
                        ("open_message_sent", "Open handshake sent"),
                        ("opened_response_received", "Opened handshake recv"),
                    ],
                )


def generate_sample_log(
    output_path: str,
    conversation_id: str = "e2e-genesys-replay-001",
    organization_id: str = "22352111-6076-492a-8163-514a00723975",
) -> None:
    """Write a minimal traffic file: single ``open`` message (AudioHook)."""
    now = datetime.now(timezone.utc).isoformat()
    msg_id = "e2e-genesys-open-001"
    open_message: Dict[str, Any] = {
        "version": "2",
        "id": msg_id,
        "type": "open",
        "seq": 1,
        "position": "PT0.0S",
        "parameters": {
            "organizationId": organization_id,
            "conversationId": conversation_id,
            "participant": {
                "id": conversation_id,
                "ani": "tel:+15551234567",
                "aniName": "",
                "dnis": "+15559876543",
            },
            "media": [
                {
                    "type": "audio",
                    "format": "PCMU",
                    "channels": ["external"],
                    "rate": 8000,
                }
            ],
            "language": "en-us",
            "inputVariables": {},
        },
        "serverseq": 0,
    }
    log_entries = [
        {
            "timestamp": now,
            "direction": "incoming",
            "type": "websocket",
            "data": open_message,
        }
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2)
    logger.info("Generated Genesys traffic log at %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genesys AudioHook WebSocket replay / generate"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=None,
        help="Path to genesys_traffic.json (required for replay)",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "generate"],
        default="replay",
    )
    parser.add_argument(
        "--output",
        default="data/e2e_voice/genesys_traffic.json",
        help="Output path for generate mode",
    )
    parser.add_argument(
        "--conversation-id",
        default="e2e-genesys-replay-001",
        help="conversationId for generate mode",
    )
    parser.add_argument(
        "--organization-id",
        default="22352111-6076-492a-8163-514a00723975",
        help="organizationId for generate mode",
    )
    parser.add_argument(
        "--rasa-url",
        default="http://localhost:5005",
    )
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    if args.mode == "generate":
        generate_sample_log(
            args.output,
            conversation_id=args.conversation_id,
            organization_id=args.organization_id,
        )
        return

    if not args.log_file:
        parser.error("log_file is required for replay mode")

    replay = GenesysReplay(
        args.rasa_url,
        args.log_file,
        timeout=args.timeout,
        delay=args.delay,
    )
    asyncio.run(replay.replay_genesys_session())


if __name__ == "__main__":
    main()
