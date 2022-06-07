# FIXME: aio_pika 7.0 comes with types, we might want to upgrade
# https://github.com/mosquito/aio-pika/blob/master/CHANGELOG.md#700
from aio_pika import patterns, pool
from aio_pika.channel import Channel
from aio_pika.connection import Connection, connect
from aio_pika.exceptions import AMQPException, MessageProcessError
from aio_pika.exchange import Exchange, ExchangeType
from aio_pika.message import DeliveryMode, IncomingMessage, Message
from aio_pika.queue import Queue
from aio_pika.robust_channel import RobustChannel
from aio_pika.robust_connection import RobustConnection, connect_robust
from aio_pika.robust_exchange import RobustExchange
from aio_pika.robust_queue import RobustQueue
from aio_pika.version import (
    __author__,
    __version__,
    author_info,
    package_info,
    package_license,
    version_info,
)

__all__ = (
    "__author__",
    "__version__",
    "author_info",
    "connect",
    "connect_robust",
    "package_info",
    "package_license",
    "patterns",
    "pool",
    "version_info",
    "AMQPException",
    "Channel",
    "Connection",
    "DeliveryMode",
    "Exchange",
    "ExchangeType",
    "IncomingMessage",
    "Message",
    "MessageProcessError",
    "Queue",
    "RobustChannel",
    "RobustConnection",
    "RobustExchange",
    "RobustQueue",
)
