from typing import Any, Dict, Iterable, List, Optional, Text, Union

from sanic.__version__ import __version__
from sanic.app import Sanic as _BaseSanic
from sanic.blueprints import Blueprint as _BaseBlueprint
from sanic.constants import HTTPMethod
from sanic.mixins.routes import RouteWrapper
from sanic.request import Request
from sanic.response import HTTPResponse, html, json, text

__all__ = [
    "__version__",
    "Sanic",
    "Blueprint",
    "HTTPMethod",
    "HTTPResponse",
    "Request",
    "html",
    "json",
    "text",
]

class Sanic(_BaseSanic):
    def stop(self) -> None: ...
    def exception(self, *exceptions: Exception, apply: bool = True) -> RouteWrapper: ...

class Blueprint(_BaseBlueprint):
    def register(self, app: Sanic, options: Dict[Text, Any]) -> None: ...
    # FIXME: Sanic uses a lazy() untyped decorator
    def route(
        self,
        uri: Text,
        methods: Optional[Iterable[Text]] = None,
        host: Optional[Union[Text, List[Text]]] = None,
        strict_slashes: Optional[bool] = None,
        stream: bool = False,
        version: Optional[Union[int, Text, float]] = None,
        name: Optional[Text] = None,
        ignore_body: bool = False,
        apply: bool = True,
        subprotocols: Optional[List[Text]] = None,
        websocket: bool = False,
        unquote: bool = False,
        static: bool = False,
        version_prefix: Text = "/v",
        error_format: Optional[Text] = None,
        **ctx_kwargs: Any,
    ) -> RouteWrapper: ...
