from typing import Callable, Text, TypeVar

from socketio.asgi import ASGIApp
from socketio.asyncio_aiopika_manager import AsyncAioPikaManager
from socketio.asyncio_client import AsyncClient
from socketio.asyncio_manager import AsyncManager
from socketio.asyncio_namespace import AsyncNamespace, AsyncClientNamespace
from socketio.asyncio_redis_manager import AsyncRedisManager
from socketio.asyncio_server import AsyncServer as _BaseAsyncServer
from socketio.base_manager import BaseManager
from socketio.client import Client
from socketio.kafka_manager import KafkaManager
from socketio.kombu_manager import KombuManager
from socketio.middleware import WSGIApp, Middleware
from socketio.namespace import Namespace, ClientNamespace
from socketio.pubsub_manager import PubSubManager
from socketio.redis_manager import RedisManager
from socketio.server import Server
from socketio.tornado import get_tornado_handler
from socketio.zmq_manager import ZmqManager

_all__ = [
    "ASGIApp",
    "AsyncAioPikaManager",
    "AsyncClient",
    "AsyncClientNamespace",
    "AsyncManager",
    "AsyncNamespace",
    "AsyncRedisManager",
    "AsyncServer",
    "BaseManager",
    "Client",
    "ClientNamespace",
    "get_tornado_handler",
    "KafkaManager",
    "KombuManager",
    "Middleware",
    "Namespace",
    "PubSubManager",
    "RedisManager",
    "Server",
    "WSGIApp",
    "ZmqManager",
]

Handler = TypeVar("Handler")
SetHandler = Callable[[Handler], Handler]

class AsyncServer(_BaseAsyncServer):
    def on(
        self, event: Text, handler: Handler = None, namespace: Text = None
    ) -> SetHandler: ...
