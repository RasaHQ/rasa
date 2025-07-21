import pytest
import requests
from websockets.client import connect


@pytest.mark.asyncio
async def test_jambonz_webhook_and_websocket():
    """Test webhook endpoint and verify websocket endpoint exists."""
    server_location = "http://localhost:5005"
    webhook_url = f"{server_location}/webhooks/jambonz_stream/webhook"
    response = requests.post(webhook_url, json={})
    assert response.status_code == 200

    # Get websocket URL from response
    json_response = response.json()
    assert isinstance(json_response, list) and len(json_response) > 0
    websocket_url = json_response[0]["url"]
    assert websocket_url.startswith(("ws://", "wss://"))

    # Try to connect to the websocket endpoint
    try:
        async with connect(websocket_url) as websocket:
            # Connection successful - endpoint exists
            assert websocket is not None
    except Exception as e:
        pytest.fail(f"Failed to connect to {websocket_url}: {e}")
