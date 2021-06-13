from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Dict, Optional, Text
import webbrowser

import nacl.encoding
from nacl.public import PrivateKey, SealedBox
import requests
from sanic import Sanic
import socketio

from rasa.constants import CONNECT_AUTH_PATH
from rasa.shared.exceptions import FileIOException, FileNotFoundException
import rasa.shared.utils.io
from rasa.shared.utils.cli import print_success
from rasa.utils.endpoints import EndpointConfig

# base_url = "http://localhost:3000"
base_url = "https://bridge.rasa.com"

logger = logging.getLogger(__name__)


@dataclass
class Webhook:
    encryption_key: PrivateKey
    token: str


@dataclass
class DeviceInfo:
    activation_url: str
    device_code: str


def url_for_webhook(webhook: Webhook):
    return f"{base_url}/h/{webhook.token}"


def exchange_device_for_access_token(
    device_code: Text, poll_interval: int = 1, timeout: int = 300
) -> Optional[Dict[Text, Any]]:
    exchange_url = base_url + "/auth/token"
    payload = {"device_code": device_code}

    passed_time = 0
    while passed_time < timeout:
        response = requests.post(exchange_url, json=payload)
        if response.status_code != 200:
            logger.debug(
                f"Retrying to exchange device code for access token."
                f"Last Response ({response.status_code}): "
                f"{response.text}"
            )
            time.sleep(poll_interval)
            passed_time += poll_interval
            continue

        jwt = response.json()
        logger.debug(f"Successfully exchanged device code for access token")
        return jwt

    logger.error(
        "Failed to exchange device code for access token. Operation "
        "timed out, please retry."
    )
    return None


def decode_base64(data: Text) -> Text:
    return nacl.encoding.Base64Encoder.decode(data.encode()).decode()


def decrypt_webhook(encrypted: Text, webhook: Webhook) -> Dict[Text, Any]:
    encrypted_binary = nacl.encoding.Base64Encoder.decode(encrypted.encode())
    unseal_box = SealedBox(webhook.encryption_key)
    # decrypt the received message
    plaintext = unseal_box.decrypt(encrypted_binary)
    request = json.loads(plaintext)
    request["body"] = decode_base64(request["body"])
    request["query"] = decode_base64(request["query"])
    logger.debug(f"Decoded request: {request}")
    return request


def webhook_from_endpoint(endpoint: EndpointConfig):
    key = endpoint.kwargs["secret"].encode()
    private_key = PrivateKey(key, nacl.encoding.Base64Encoder)
    return Webhook(private_key, token=endpoint.token or "")


async def retrieve_webhook_calls(
    access_token, webhook: Webhook, app: Sanic, server: str
):
    sio = socketio.AsyncClient(reconnection=True)
    headers = auth_header(access_token)

    @sio.on("webhook", namespace="/local")
    async def trigger_webhook(data):
        logger.debug(f"Received webhook socket trigger.")
        decrypted = decrypt_webhook(data["payload"], webhook)
        logger.debug(f"Decrypted data: {decrypted}")

        if decrypted["path"].startswith("/webhooks/"):
            url = f"http://{server}{decrypted['path']}?{decrypted['query']}"
            logger.debug(f"Sending request to {url}")
            response = await app.test_client._local_request(
                decrypted["method"],
                url,
                data=decrypted["body"],
                headers=decrypted["headers"],
            )
            logger.debug(f"Response from {url}: {response.status}")
            return json.dumps({"status": response.status, "body": response.text})
        else:
            logger.error("Skipped webhook trigger due to wrong path.")

    @sio.on("connect", namespace="/local")
    async def connect():
        await sio.emit("subscribe", {"token": webhook.token}, namespace="/local")
        logger.debug(f"Subscribed to webhook.")

    await sio.connect(
        f"{base_url}/ws",
        socketio_path="/ws/socket.io",
        headers=headers,
        namespaces=["/local"],
    )


def get_webhook_info(access_token) -> Optional[Webhook]:
    headers = auth_header(access_token)
    keypair = PrivateKey.generate()
    public_key = keypair.public_key.encode(nacl.encoding.Base64Encoder).decode()
    webhook_url = base_url + "/api/webhooks"
    payload = {"public": public_key}

    response = requests.post(webhook_url, json=payload, headers=headers)
    response.raise_for_status()
    webhook_info = response.json()

    return Webhook(encryption_key=keypair, token=webhook_info["token"])


def get_activation_info() -> DeviceInfo:
    device_url = base_url + "/auth/device"

    response = requests.get(device_url)
    response.raise_for_status()
    device_info = response.json()

    return DeviceInfo(
        activation_url=device_info["activation_url"],
        device_code=device_info["device_code"],
    )


def get_access_token() -> Dict[Text, Any]:
    device_info = get_activation_info()
    webbrowser.open(device_info.activation_url)
    return exchange_device_for_access_token(device_info.device_code)


def auth_header(access_token: Dict[Text, Any]) -> Dict[Text, Any]:
    return {"Authorization": f"Bearer {access_token['id_token']}"}


def persist_token(access_token: Dict[Text, Any]):
    rasa.shared.utils.io.dump_obj_as_json_to_file(CONNECT_AUTH_PATH, access_token)


def retrieve_token_from_disk(
    path: str = CONNECT_AUTH_PATH,
) -> Optional[Dict[Text, Any]]:
    try:
        return rasa.shared.utils.io.read_json_file(path)
    except (FileIOException, FileNotFoundException) as e:
        return None


def retrieve_access_token() -> Optional[str]:
    access_token = retrieve_token_from_disk()
    if not access_token:
        access_token = get_access_token()
        persist_token(access_token)
    return access_token


def create_webhook() -> Optional[Webhook]:
    access_token = retrieve_access_token()
    if not access_token:
        logger.error("Failed to retrieve access token!")
        return None

    webhook = get_webhook_info(access_token)
    return webhook
