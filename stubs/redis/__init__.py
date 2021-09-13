from typing import Text, List, overload, Optional, Union, Mapping, Literal

from redis import ConnectionPool
from typing_extensions import Protocol

# We should switch to https://pypi.org/project/types-redis/ once
# https://github.com/python/typeshed/issues/5065 is fixed.
class StrictRedis(Protocol):
    @overload
    def __init__(
        self,
        host: Text = ...,
        port: int = ...,
        db: int = ...,
        password: Optional[Text] = ...,
        socket_timeout: Optional[float] = ...,
        socket_connect_timeout: Optional[float] = ...,
        socket_keepalive: Optional[bool] = ...,
        socket_keepalive_options: Optional[Mapping[str, Union[int, str]]] = ...,
        connection_pool: Optional[ConnectionPool] = ...,
        unix_socket_path: Optional[Text] = ...,
        encoding: Text = ...,
        encoding_errors: Text = ...,
        charset: Optional[Text] = ...,
        errors: Optional[Text] = ...,
        decode_responses: Literal[False] = ...,
        retry_on_timeout: bool = ...,
        ssl: bool = ...,
        ssl_keyfile: Optional[Text] = ...,
        ssl_certfile: Optional[Text] = ...,
        ssl_cert_reqs: Optional[Union[str, int]] = ...,
        ssl_ca_certs: Optional[Text] = ...,
        ssl_check_hostname: bool = ...,
        max_connections: Optional[int] = ...,
        single_connection_client: bool = ...,
        health_check_interval: float = ...,
        client_name: Optional[Text] = ...,
        username: Optional[Text] = ...,
    ) -> None:
        ...

    def keys(self, pattern: Text) -> List[Text]:
        ...
