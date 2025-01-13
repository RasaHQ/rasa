from sanic import Sanic
from sanic.exceptions import ServerError
from sanic.handlers import ErrorHandler
from sanic.request import Request


# TODO: remove custom handler when upgrading to sanic >= 24
#       the underlying issue https://github.com/sanic-org/sanic/issues/2572
#       has been fixed in sanic 24
class IgnoreWSServerErrorHandler(ErrorHandler):
    @staticmethod
    def log(request: Request, exception: Exception) -> None:
        try:
            if (
                request.url.startswith("ws")
                and isinstance(exception, ServerError)
                and exception.args
                and (
                    exception.args[0]
                    == "Invalid response type None (need HTTPResponse)"
                )
            ):
                # in case we are in a websocket connection, we don't want to log the
                # the error, as this is a bug in sanic
                return
        except Exception:
            pass
        ErrorHandler.log(request, exception)  # type: ignore


def register_custom_sanic_error_handler(app: Sanic) -> None:
    app.error_handler = IgnoreWSServerErrorHandler()
