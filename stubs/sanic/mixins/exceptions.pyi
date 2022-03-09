from sanic.mixins.routes import RouteWrapper
from typing_extensions import Protocol


class ExceptionMixin(Protocol):

    def exception(self, *exceptions: Exception, apply: bool =True) -> RouteWrapper:
        ...
