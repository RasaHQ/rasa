from typing import Any, Dict, Optional

import structlog
from socketio import AsyncClient, AsyncServer  # type: ignore[attr-defined]
from socketio.exceptions import ConnectionRefusedError

from rasa.model_manager.runner_service import BotSession
from rasa.model_manager.studio_jwt_auth import (
    UserToServiceAuthenticationError,
    authenticate_user_to_service,
)

structlogger = structlog.get_logger()


# A simple in-memory store for active chat connections to studio frontend
socket_proxy_clients = {}


async def socketio_websocket_traffic_wrapper(
    sio_server: AsyncServer,
    running_bots: Dict[str, BotSession],
    sid: str,
    auth: Optional[Dict],
) -> bool:
    """Wrapper for bridging the user chat websocket and the bot server."""
    auth_token = auth.get("token") if auth else None

    if auth_token is None:
        structlogger.error("model_runner.user_no_token", sid=sid)
        raise ConnectionRefusedError("model_runner.user_no_token")

    try:
        authenticate_user_to_service(auth_token)
        structlogger.debug("model_runner.user_authenticated_successfully", sid=sid)
    except UserToServiceAuthenticationError as error:
        structlogger.error(
            "model_runner.user_authentication_failed", sid=sid, error=str(error)
        )
        raise ConnectionRefusedError("model_runner.user_authentication_failed")

    deployment_id = auth.get("deployment_id") if auth else None

    if deployment_id is None:
        structlogger.error("model_runner.bot_no_deployment_id", sid=sid)
        raise ConnectionRefusedError("model_runner.bot_no_deployment_id")

    bot = running_bots.get(deployment_id)
    if bot is None:
        structlogger.error("model_runner.bot_not_found", deployment_id=deployment_id)
        raise ConnectionRefusedError("model_runner.bot_not_found")

    if not bot.is_alive():
        structlogger.error("model_runner.bot_not_alive", deployment_id=deployment_id)
        raise ConnectionRefusedError("model_runner.bot_not_alive")

    client = await create_bridge_client(
        sio_server, bot.internal_url, sid, deployment_id
    )

    if client.sid is not None:
        structlogger.debug(
            "model_runner.bot_connection_established", deployment_id=deployment_id
        )
        socket_proxy_clients[sid] = client
        return True
    else:
        structlogger.error(
            "model_runner.bot_connection_failed", deployment_id=deployment_id
        )
        raise ConnectionRefusedError("model_runner.bot_connection_failed")


def create_bridge_server(
    sio_server: AsyncServer, running_bots: Dict[str, BotSession]
) -> None:
    """Create handlers for the socket server side.

    Forwards messages coming from the user to the bot.
    """

    @sio_server.on("connect")
    async def socketio_websocket_traffic(
        sid: str, environ: Dict, auth: Optional[Dict]
    ) -> bool:
        """Bridge websockets between user chat socket and bot server."""
        return await socketio_websocket_traffic_wrapper(
            sio_server, running_bots, sid, auth
        )

    @sio_server.on("disconnect")
    async def disconnect(sid: str) -> None:
        """Disconnect the bot connection."""
        structlogger.debug("model_runner.bot_disconnect", sid=sid)
        if sid in socket_proxy_clients:
            await socket_proxy_clients[sid].disconnect()
            del socket_proxy_clients[sid]

    @sio_server.on("*")
    async def handle_message(event: str, sid: str, data: Dict[str, Any]) -> None:
        """Bridge messages between user and bot.

        Both incoming user messages to the bot_url and
        bot responses sent back to the client need to
        happen in parallel in an async way.
        """
        client = socket_proxy_clients.get(sid)
        if client is None:
            structlogger.error("model_runner.bot_not_connected", sid=sid)
            return

        await client.emit(event, data)


async def create_bridge_client(
    sio_server: AsyncServer, url: str, sid: str, deployment_id: str
) -> AsyncClient:
    """Create a new socket bridge client.

    Forwards messages coming from the bot to the user.
    """
    client = AsyncClient()

    await client.connect(url)

    @client.event  # type: ignore[misc]
    async def session_confirm(data: Dict[str, Any]) -> None:
        structlogger.debug(
            "model_runner.bot_session_confirmed", deployment_id=deployment_id
        )
        await sio_server.emit("session_confirm", room=sid)

    @client.event  # type: ignore[misc]
    async def bot_message(data: Dict[str, Any]) -> None:
        structlogger.debug("model_runner.bot_message", deployment_id=deployment_id)
        await sio_server.emit("bot_message", data, room=sid)

    @client.event  # type: ignore[misc]
    async def error(data: Dict[str, Any]) -> None:
        structlogger.debug(
            "model_runner.bot_error", deployment_id=deployment_id, data=data
        )
        await sio_server.emit("error", data, room=sid)

    @client.event  # type: ignore[misc]
    async def tracker(data: Dict[str, Any]) -> None:
        await sio_server.emit("tracker", data, room=sid)

    @client.event  # type: ignore[misc]
    async def disconnect() -> None:
        structlogger.debug(
            "model_runner.bot_connection_closed", deployment_id=deployment_id
        )
        await sio_server.emit("disconnect", room=sid)

    @client.event  # type: ignore[misc]
    async def connect_error() -> None:
        structlogger.error(
            "model_runner.bot_connection_error", deployment_id=deployment_id
        )
        await sio_server.emit("disconnect", room=sid)

    return client
