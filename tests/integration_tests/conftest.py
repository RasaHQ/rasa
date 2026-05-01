import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests


def send_message_to_rasa_server(
    server_location: str, message: str = "message", sender_id: Optional[str] = None
) -> (str, List[Dict[str, Any]]):  # type: ignore
    """Send a message to the REST channel and return the sender id."""
    if not sender_id:
        sender_id = str(uuid.uuid4())
    response = requests.post(
        f"{server_location}/webhooks/rest/webhook",
        json={"sender": sender_id, "message": message},
    )
    json_response = response.json()
    return sender_id, json_response


def send_streaming_message_to_rasa_server(
    server_location: str,
    message: str,
    sender_id: Optional[str] = None,
    verify: Any = True,
    timeout: int = 30,
) -> Tuple[str, List[Dict[str, Any]]]:
    """POST to ``/webhooks/rest/webhook?stream=true`` and collect all SSE chunks.

    The REST SSE endpoint writes newline-delimited JSON objects (one per line)
    and closes the connection when the action completes.  This helper reads the
    response as a stream and parses each non-empty line as JSON.

    Args:
        server_location: Base URL of the Rasa server, e.g. ``http://localhost:5012``.
        message: Text of the user message to send.
        sender_id: Conversation sender ID.  A UUID is generated if omitted.
        verify: Passed directly to ``requests.post`` as the ``verify`` argument.
            Pass a path to a CA bundle to validate a self-signed TLS certificate,
            or ``False`` to skip verification (not recommended in production).
        timeout: Request timeout in seconds.

    Returns:
        A ``(sender_id, events)`` tuple where *events* is the ordered list of
        JSON dicts received over the stream.
    """
    if not sender_id:
        sender_id = str(uuid.uuid4())

    with requests.post(
        f"{server_location}/webhooks/rest/webhook",
        params={"stream": "true"},
        json={"sender": sender_id, "message": message},
        stream=True,
        verify=verify,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        events: List[Dict[str, Any]] = []
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line.strip()
            if line:
                events.append(json.loads(line))

    return sender_id, events


def get_conversation_tracker(server_location: str, conversation_id: str) -> dict:
    """Gets the tracker for a given conversation ID from the Rasa server.

    Args:
        server_location (str): The base URL of the Rasa server (e.g., "http://localhost:5005").
        conversation_id (str): The ID of the conversation.
        api_key (str): The API key for authentication (if required).

    Returns:
        dict: The tracker as a dictionary, or None if there was an error.
    """
    url = f"{server_location}/conversations/{conversation_id}/tracker"
    params = {
        "include_events": "AFTER_RESTART",  # Include events after the last restart
        "until": None,  # Get all events up to the present
    }

    auth_token = os.getenv("AUTH_TOKEN")
    if auth_token:
        params["token"] = auth_token

    try:
        response = requests.get(
            url, params=params, headers={"Accept": "application/json"}
        )
        # Raise an exception for non-200 status codes
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting tracker store: {e}")
        return None


def was_enterprise_search_policy_used(tracker_json: Dict[str, Any]) -> bool:
    """Checks if the EnterpriseSearchPolicy was used in the conversation.

    Args:
        tracker_json (dict): The tracker JSON data.

    Returns:
        bool: True if the EnterpriseSearchPolicy was used, otherwise False.
    """
    events = tracker_json["events"]
    for evt in reversed(events):
        if evt.get("name") == "action_send_text":
            return True

        # collect until the last user message
        if evt.get("event") == "user":
            return False

    return False


def extract_enterprise_search_results(tracker_json: Dict[str, Any]) -> list:
    """Extracts the search results from the EnterpriseSearchPolicy response
    in the tracker JSON.

    Args:
        tracker_json (dict): The tracker JSON data.

    Returns:
        list: A list of search results (strings) if found, otherwise an empty list.
    """
    events = tracker_json["events"]
    for evt in reversed(events):
        if evt.get("name") == "action_send_text":
            return evt["metadata"]["message"]["search_results"]

    return ["No search results found."]
