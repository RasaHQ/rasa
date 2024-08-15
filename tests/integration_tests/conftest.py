import uuid
from typing import Any, Dict, List, Optional

import requests


def send_message_to_rasa_server(
    server_location: str, message: str = "message", sender_id: Optional[str] = None
) -> (str, List[Dict[str, Any]]):
    """Send a message to the REST channel and return the sender id."""
    if not sender_id:
        sender_id = str(uuid.uuid4())
    response = requests.post(
        f"{server_location}/webhooks/rest/webhook",
        json={"sender": sender_id, "message": message},
    )
    json_response = response.json()
    return sender_id, json_response
