import json
import uuid
from typing import Any, Dict, List

import requests


def send_message_to_rasa_server(
    server_location: str, message: str = "message"
) -> (str, List[Dict[str, Any]]):
    """Send a message to the REST channel and return the sender id."""
    sender_id = str(uuid.uuid4())
    response = requests.post(
        f"{server_location}/webhooks/rest/webhook",
        json={"sender": sender_id, "message": message},
    )
    json_response = response.json()
    return sender_id, json_response
