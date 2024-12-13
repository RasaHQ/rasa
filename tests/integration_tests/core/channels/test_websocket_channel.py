import asyncio
from tests.integration_tests.core.channels.socketio_client import RasaSocketIOClient


async def test_send_message():
    client = RasaSocketIOClient(url="http://localhost:5005")
    await client.connect_to_server()
    assert client.sio.connected

    # Send a message and wait for a response
    await client.send_message("I want to transfer money!")
    await asyncio.sleep(2)  # Wait for response
    # Assert that the bot responded
    assert len(client.bot_responses) > 0
    assert "transfer money" in client.bot_responses[0].lower()
    assert client.sio.connected

    await client.sio.disconnect()
