from typing import Union

# mypy check fails here but it actually successfully loads the initial module
# so it's probably an internal issue of mypy with no repercussions
from aio_pika.robust_channel import RobustChannel as AioPikaRobustChannel  # type: ignore[attr-defined]
from aio_pika.robust_exchange import RobustExchange
from aio_pika.exchange import ExchangeType
from aio_pika.types import TimeoutType

class RobustChannel(AioPikaRobustChannel):
    async def declare_exchange(
        self,
        name: str,
        type: Union[ExchangeType, str] = ExchangeType.DIRECT,
        durable: bool = None,
        auto_delete: bool = False,
        internal: bool = False,
        passive: bool = False,
        arguments: dict = None,
        timeout: TimeoutType = None,
        robust: bool = True,
    ) -> RobustExchange: ...
