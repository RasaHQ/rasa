"""
Test utility for replaying AudioCodes WebSocket traffic
WebSocket Traffic Replay Tool
Replays captured AudioCodes WebSocket traffic for testing Rasa
"""

import argparse
import asyncio
import json
import logging

from aiohttp import ClientSession, WSMsgType

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebSocketReplay:
    def __init__(self, rasa_url, log_file, timeout=5, delay=1):
        self.rasa_url = rasa_url
        self.log_file = log_file
        self.timeout = timeout
        self.delay = delay
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
        logger.info(f"Loading traffic from {self.log_file}")

        try:
            with open(self.log_file, "r") as f:
                traffic_log = json.load(f)
        except FileNotFoundError:
            logger.error(f"File not found: {self.log_file}")
            return
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in log file: {e}")
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

        # Connect to Rasa WebSocket
        ws_url = self.rasa_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/webhooks/audiocodes_stream/websocket"

        logger.info(f"Connecting to Rasa at {ws_url}")

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

                        # Check connection state before sending
                        if ws.closed:
                            logger.debug(
                                f"Connection closed before sending message "
                                f"{i} ({message_type})"
                            )
                            self.connection_state["errors"].append(
                                {
                                    "message_index": i,
                                    "message_type": message_type,
                                    "error": "Connection closed before sending",
                                }
                            )
                            break

                        # Send the message
                        try:
                            await ws.send_json(data)
                            self.connection_state["messages_sent"] += 1

                            # Update state tracking
                            if message_type == "session.initiate":
                                self.connection_state["session_initiated"] = True
                            elif message_type == "activities":
                                activities = data.get("activities", [])
                                if any(a.get("name") == "start" for a in activities):
                                    self.connection_state["activities_start_sent"] = (
                                        True
                                    )
                        except Exception as send_error:
                            error_str = str(send_error)
                            error_type = type(send_error).__name__

                            # Check if this is a graceful connection closure
                            is_graceful_closure = (
                                "closing transport" in error_str.lower()
                                or "connection closed" in error_str.lower()
                                or error_type
                                in (
                                    "ConnectionResetError",
                                    "ClientConnectionResetError",
                                )
                            )

                            if is_graceful_closure:
                                logger.debug(
                                    f"Connection closed by Rasa while sending "
                                    f"message {i} ({message_type}) - this is "
                                    f"expected when conversation ends"
                                )
                                self.connection_state["errors"].append(
                                    {
                                        "message_index": i,
                                        "message_type": message_type,
                                        "error": (
                                            f"Connection closed gracefully: "
                                            f"{error_str}"
                                        ),
                                        "graceful": True,
                                    }
                                )
                            else:
                                logger.debug(
                                    f"Error sending message {i} "
                                    f"({message_type}): {send_error}"
                                )
                                self.connection_state["errors"].append(
                                    {
                                        "message_index": i,
                                        "message_type": message_type,
                                        "error": str(send_error),
                                        "graceful": False,
                                    }
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
                                        f"WebSocket closed by server "
                                        f"(code: {close_code})"
                                    )
                                    self.connection_state["errors"].append(
                                        {
                                            "message_index": i,
                                            "message_type": message_type,
                                            "error": (
                                                f"Connection closed by server "
                                                f"(code: {close_code})"
                                            ),
                                        }
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

                    # Print summary
                    self._print_session_summary()

                    logger.info("✓ Replay complete")

                    # Close connection gracefully if still open
                    if not ws.closed:
                        try:
                            await ws.close()
                            logger.info("✓ Connection closed gracefully")
                        except Exception as close_error:
                            logger.warning(f"Error closing connection: {close_error}")

            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                self.connection_state["errors"].append(
                    {"error_type": type(e).__name__, "error_message": str(e)}
                )

                # Print summary even on error
                self._print_session_summary()

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
        # Messages that typically don't have responses
        no_response_types = {"userStream.chunk"}
        return message_type not in no_response_types

    def _print_session_summary(self) -> None:
        """Print a summary of the session"""
        logger.info("\n" + "=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.debug(
            f"Messages sent:         " f"{self.connection_state['messages_sent']}"
        )
        logger.debug(
            f"Messages received:    " f"{self.connection_state['messages_received']}"
        )
        logger.info(
            f"Session initiated:    " f"{self.connection_state['session_initiated']}"
        )
        logger.info(
            f"Session accepted:     " f"{self.connection_state['session_accepted']}"
        )
        logger.info(
            f"Activities start sent: "
            f"{self.connection_state['activities_start_sent']}"
        )

        if self.connection_state["errors"]:
            graceful_errors = [
                e for e in self.connection_state["errors"] if e.get("graceful", False)
            ]
            actual_errors = [
                e
                for e in self.connection_state["errors"]
                if not e.get("graceful", False)
            ]

            if graceful_errors:
                logger.info(
                    f"\nGraceful connection closures: " f"{len(graceful_errors)}"
                )
                for error in graceful_errors:
                    msg_idx = error.get("message_index", "?")
                    msg_type = error.get("message_type", "unknown")
                    error_msg = error.get("error", "Connection closed")
                    logger.info(f"  - Message {msg_idx} ({msg_type}): {error_msg}")

            if actual_errors:
                logger.warning(f"\nErrors encountered: {len(actual_errors)}")
                for error in actual_errors:
                    msg_idx = error.get("message_index", "?")
                    msg_type = error.get("message_type", "unknown")
                    error_msg = error.get("error", "Unknown error")
                    logger.warning(f"  - Message {msg_idx} ({msg_type}): {error_msg}")
            elif not graceful_errors:
                logger.info("\nNo errors encountered")
        else:
            logger.info("\nNo errors encountered")

        logger.info("=" * 60 + "\n")

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


def main():
    parser = argparse.ArgumentParser(description="WebSocket Traffic Replay Tool")
    parser.add_argument("log_file", help="Path to audiocodes_traffic.json file")
    parser.add_argument(
        "--rasa-url",
        default="http://localhost:5005",
        help="Rasa server URL (default: http://localhost:5005)",
    )
    parser.add_argument(
        "--mode",
        choices=["replay", "analyze"],
        default="replay",
        help="Mode: replay messages or analyze traffic",
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

    replay = WebSocketReplay(
        args.rasa_url, args.log_file, timeout=args.timeout, delay=args.delay
    )

    if args.mode == "replay":
        asyncio.run(replay.replay_websocket_session())
    elif args.mode == "analyze":
        asyncio.run(replay.analyze_traffic())


if __name__ == "__main__":
    main()
