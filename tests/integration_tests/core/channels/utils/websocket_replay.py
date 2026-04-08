"""
Test utility for replaying AudioCodes WebSocket traffic.

Replays captured AudioCodes WebSocket traffic for testing Rasa.
Supports generate mode to create a minimal traffic file for CI.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone

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


class WebSocketReplay:
    def __init__(self, rasa_url, log_file, timeout=5, delay=1):
        self.rasa_url = rasa_url
        self.log_file = log_file
        self.timeout = timeout
        self.delay = delay
        self._reset_connection_state()

    def _reset_connection_state(self) -> None:
        """Reset per-session counters so a second replay run starts clean."""
        self.connection_state = {
            "connected": False,
            "session_initiated": False,
            "session_accepted": False,
            "activities_start_sent": False,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": [],
        }

    async def replay_websocket_session(self):
        """Replay a complete WebSocket session from captured traffic"""
        self._reset_connection_state()
        logger.info("Loading traffic from %s", self.log_file)
        traffic_log = load_traffic_log(self.log_file)
        if traffic_log is None:
            return

        # Filter for WebSocket messages
        incoming = [
            entry
            for entry in traffic_log
            if entry["direction"] == "incoming" and entry["type"] == "websocket"
        ]

        outgoing = [
            entry
            for entry in traffic_log
            if entry["direction"] == "outgoing" and entry["type"] == "websocket"
        ]

        logger.info(
            f"Found {len(incoming)} incoming and {len(outgoing)} "
            f"outgoing WebSocket messages"
        )

        # Check if activities message with name="start" exists
        for entry in incoming:
            data = entry.get("data", {})
            if data.get("type") == "activities":
                activities = data.get("activities", [])
                for activity in activities:
                    if activity.get("name") == "start":
                        break

        if not incoming:
            logger.debug("No WebSocket messages found in log file")
            return

        # Validate message flow
        validation_result = self._validate_message_flow(incoming)
        if not validation_result["valid"]:
            logger.warning(
                f"Message flow validation warnings: " f"{validation_result['warnings']}"
            )
            for warning in validation_result["warnings"]:
                logger.warning(f"  - {warning}")

        ws_url = build_websocket_url(
            self.rasa_url, "webhooks/audiocodes_stream/websocket"
        )
        logger.info("Connecting to Rasa at %s", ws_url)

        async with ClientSession() as session:
            try:
                async with session.ws_connect(ws_url) as ws:
                    self.connection_state["connected"] = True
                    logger.info("✓ WebSocket connected")

                    # Send all incoming messages
                    for i, entry in enumerate(incoming, 1):
                        data = entry["data"]
                        message_type = data.get("type", "unknown")

                        logger.debug(f"[{i}/{len(incoming)}] Sending: {message_type}")
                        logger.debug(f"Data: {json.dumps(data, indent=2)}")

                        if ws.closed:
                            logger.debug(
                                "Connection closed before sending message %s (%s)",
                                i,
                                message_type,
                            )
                            record_connection_closed_before_send(
                                self.connection_state["errors"], i, message_type
                            )
                            break

                        try:
                            await ws.send_json(data)
                            self.connection_state["messages_sent"] += 1
                            if message_type == "session.initiate":
                                self.connection_state["session_initiated"] = True
                            elif message_type == "activities":
                                activities = data.get("activities", [])
                                start_sent = any(
                                    a.get("name") == "start" for a in activities
                                )
                                if start_sent:
                                    self.connection_state["activities_start_sent"] = (
                                        True
                                    )
                        except Exception as send_error:
                            if is_graceful_connection_error(send_error):
                                logger.debug(
                                    "Connection closed by Rasa while sending "
                                    "message %s (%s) - expected when conversation ends",
                                    i,
                                    message_type,
                                )
                            else:
                                logger.debug(
                                    "Error sending message %s (%s): %s",
                                    i,
                                    message_type,
                                    send_error,
                                )
                            record_send_error(
                                self.connection_state["errors"],
                                i,
                                message_type,
                                send_error,
                            )
                            break

                        # Wait for response (some messages may not have responses)
                        response_type = None
                        should_wait_for_response = self._should_wait_for_response(
                            message_type
                        )

                        if should_wait_for_response:
                            try:
                                msg = await asyncio.wait_for(
                                    ws.receive(), timeout=self.timeout
                                )
                                self.connection_state["messages_received"] += 1

                                if msg.type == WSMsgType.TEXT:
                                    response = json.loads(msg.data)
                                    response_type = response.get("type", "unknown")
                                    logger.info(
                                        f"← Response: "
                                        f"{json.dumps(response, indent=2)}"
                                    )

                                    # Update state tracking
                                    if response_type == "session.accepted":
                                        self.connection_state["session_accepted"] = True

                                    # Special handling for session.initiate ->
                                    # session.accepted flow
                                    if (
                                        message_type == "session.initiate"
                                        and response_type == "session.accepted"
                                    ):
                                        logger.debug(
                                            "Session initiate accepted, "
                                            "proceeding with next message"
                                        )
                                elif msg.type == WSMsgType.BINARY:
                                    logger.debug(
                                        f"← Binary response: " f"{len(msg.data)} bytes"
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
                                        message_type,
                                        close_code,
                                    )
                                    break
                            except asyncio.TimeoutError:
                                if should_wait_for_response:
                                    logger.debug(
                                        f"No response received for "
                                        f"{message_type} (timeout after "
                                        f"{self.timeout}s)"
                                    )
                        else:
                            logger.debug(
                                f"Skipping response wait for {message_type} "
                                f"(not expected)"
                            )

                        # Pause between messages
                        if i < len(incoming):  # Don't delay after last message
                            await asyncio.sleep(self.delay)

                    print_session_summary(
                        self.connection_state,
                        state_keys=[
                            ("session_initiated", "Session initiated"),
                            ("session_accepted", "Session accepted"),
                            ("activities_start_sent", "Activities start sent"),
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
                        ("session_initiated", "Session initiated"),
                        ("session_accepted", "Session accepted"),
                        ("activities_start_sent", "Activities start sent"),
                    ],
                )

    def _validate_message_flow(self, incoming: list) -> dict:
        """Validate that the message flow follows expected patterns"""
        warnings = []

        # Check for session.initiate
        session_initiate_found = False
        activities_start_found = False

        for entry in incoming:
            data = entry.get("data", {})
            msg_type = data.get("type", "unknown")

            if msg_type == "session.initiate":
                session_initiate_found = True
            elif msg_type == "activities":
                activities = data.get("activities", [])
                if any(a.get("name") == "start" for a in activities):
                    activities_start_found = True

        if session_initiate_found and not activities_start_found:
            warnings.append(
                "session.initiate found but no activities message with "
                "name='start' found"
            )

        return {"valid": len(warnings) == 0, "warnings": warnings}

    def _should_wait_for_response(self, message_type: str) -> bool:
        """Determine if we should wait for a response to this message type"""
        no_response_types = {"userStream.chunk"}
        return message_type not in no_response_types

    async def analyze_traffic(self):
        """Analyze captured traffic and show statistics"""
        logger.info(f"Analyzing traffic from {self.log_file}")

        try:
            with open(self.log_file, "r") as f:
                traffic_log = json.load(f)
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return

        # Statistics
        total = len(traffic_log)
        incoming = [e for e in traffic_log if e["direction"] == "incoming"]
        outgoing = [e for e in traffic_log if e["direction"] == "outgoing"]

        incoming_ws = [e for e in incoming if e["type"] == "websocket"]
        outgoing_ws = [e for e in outgoing if e["type"] == "websocket"]

        print("\n" + "=" * 60)
        print("TRAFFIC ANALYSIS")
        print("=" * 60)
        print(f"Total messages:        {total}")
        print(f"Incoming:              {len(incoming)}")
        print(f"  - WebSocket:         {len(incoming_ws)}")
        print(f"  - HTTP:              {len(incoming) - len(incoming_ws)}")
        print(f"Outgoing:              {len(outgoing)}")
        print(f"  - WebSocket:         {len(outgoing_ws)}")
        print(f"  - HTTP:              {len(outgoing) - len(outgoing_ws)}")
        print("=" * 60)

        # Message types
        print("\nINCOMING MESSAGE TYPES:")
        message_types = {}
        for entry in incoming_ws:
            msg_type = entry.get("data", {}).get("type", "unknown")
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        for msg_type, count in sorted(message_types.items()):
            print(f"  {msg_type}: {count}")

        print("\nOUTGOING MESSAGE TYPES:")
        message_types = {}
        for entry in outgoing_ws:
            msg_type = entry.get("data", {}).get("type", "unknown")
            message_types[msg_type] = message_types.get(msg_type, 0) + 1

        for msg_type, count in sorted(message_types.items()):
            print(f"  {msg_type}: {count}")

        print("\n" + "=" * 60)

        # Show first few messages
        print("\nFIRST 3 INCOMING MESSAGES:")
        for i, entry in enumerate(incoming_ws[:3], 1):
            print(f"\n{i}. {entry.get('data', {}).get('type', 'unknown')}")
            print(f"   {json.dumps(entry.get('data', {}), indent=2)}")

        print("\n" + "=" * 60 + "\n")


def generate_sample_log(
    output_path: str,
    conversation_id: str = "e2e-audiocodes-replay-001",
    bot_name: str = "e2e-bot-001",
) -> None:
    """Generate a minimal AudioCodes WebSocket traffic log for replay tests."""
    now = datetime.now(timezone.utc).isoformat()
    log_entries = [
        {
            "timestamp": now,
            "direction": "incoming",
            "type": "websocket",
            "data": {"type": "connection.validate"},
        },
        {
            "timestamp": now,
            "direction": "incoming",
            "type": "websocket",
            "data": {
                "type": "session.initiate",
                "conversationId": conversation_id,
                "caller": "anonymous",
                "botName": bot_name,
                "expectAudioMessages": True,
                "supportedMediaFormats": [
                    "raw/lpcm16",
                    "wav/lpcm16",
                    "raw/lpcm16_24",
                    "wav/lpcm16_24",
                    "raw/mulaw",
                    "wav/mulaw",
                ],
            },
        },
        {
            "timestamp": now,
            "direction": "incoming",
            "type": "websocket",
            "data": {
                "conversationId": conversation_id,
                "type": "activities",
                "activities": [
                    {
                        "id": "e2e-activity-start-001",
                        "timestamp": now,
                        "language": "en",
                        "type": "event",
                        "name": "start",
                        "parameters": {
                            "locale": "en",
                            "callee": "LiveHub",
                            "vaigConversationId": conversation_id,
                        },
                    }
                ],
            },
        },
    ]
    with open(output_path, "w") as f:
        json.dump(log_entries, f, indent=2)
    logger.info("Generated AudioCodes traffic log at %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket Traffic Replay Tool (AudioCodes)"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default=None,
        help="Path to audiocodes_traffic.json (required for replay/analyze)",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "analyze", "generate"],
        default="replay",
        help="Mode: replay, analyze, or generate sample traffic",
    )
    parser.add_argument(
        "--output",
        default="data/e2e_voice/audiocodes_traffic.json",
        help="Output path for generate mode",
    )
    parser.add_argument(
        "--conversation-id",
        default="e2e-audiocodes-replay-001",
        help="Conversation ID for generate mode",
    )
    parser.add_argument(
        "--bot-name",
        default="e2e-bot-001",
        help="Bot name for generate mode",
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
            conversation_id=args.conversation_id,
            bot_name=args.bot_name,
        )
        return

    if not args.log_file:
        parser.error("log_file is required for replay and analyze modes")
    replay = WebSocketReplay(
        args.rasa_url, args.log_file, timeout=args.timeout, delay=args.delay
    )

    if args.mode == "replay":
        asyncio.run(replay.replay_websocket_session())
    elif args.mode == "analyze":
        asyncio.run(replay.analyze_traffic())
