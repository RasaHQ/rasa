import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

import opentelemetry.trace
import sanic
from opentelemetry.context.context import Context
from opentelemetry.trace import Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from sanic import Request, Sanic

logger = logging.getLogger()
tracer = opentelemetry.trace.get_tracer_provider().get_tracer(__name__)

# The `TypeVar` representing the return type for a function to be wrapped.
S = TypeVar("S")
# The `TypeVar` representing the type of the argument passed to the function to be
# wrapped.
T = TypeVar("T")


def traceable_async(
    tracer: Tracer,
    attr_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]] = None,
    context_extractor: Optional[Callable[[T, Any, Any], Dict[str, Any]]] = None,
) -> Callable[[T, Any, Any], Awaitable[S]]:
    """Wrap an `async` function by tracing functionality.

    :param tracer: The `Tracer` that shall be used for tracing this function.
    :param attr_extractor: A function that is applied to the function's instance and
        the function's arguments.
    :return: The wrapped function.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> S:
            attributes = attr_extractor(*args, **kwargs) if attr_extractor else {}
            context = context_extractor(*args, **kwargs) if context_extractor else {}
            with tracer.start_as_current_span(
                f"{fn.__module__}.{fn.__name__}",
                context=context,
                kind=opentelemetry.trace.SpanKind.SERVER,
                attributes=attributes,
            ):
                return await fn(*args, **kwargs)

        return async_wrapper

    return decorator


def extract_context_for_action_call(request: Request) -> Context:
    """Extract the tracing context from a custom action request's headers"""
    propagator = TraceContextTextMapPropagator()
    context = propagator.extract(request.headers)
    return context


def extract_attrs_for_action_call(request: Request) -> Dict[str, Any]:
    """Extract the traceable attributes for a custom action request"""
    return {
        "action_name": request.json.get("next_action"),
        "sender_id": request.json.get("sender_id"),
        "message_id": request.json.get("tracker", {})
        .get("latest_message", {})
        .get("message_id"),
    }


@traceable_async(
    tracer=tracer,
    attr_extractor=extract_attrs_for_action_call,
    context_extractor=extract_context_for_action_call,
)
async def run_action(request: Request):
    return {"events": [], "responses": [{"text": "hallo from the action server"}]}


def run_app(port):
    app = Sanic("action_server")

    @app.route("/webhook", methods=["POST"])
    async def webhook(request):
        """Endpoint which processes the Rasa request to run a custom action."""
        action_response = await run_action(request)
        return sanic.response.json(action_response)

    @app.route("/health", methods=["GET"])
    async def health(request):
        """Endpoint to see status of the action server."""
        return sanic.response.text("OK")

    app.run(host="0.0.0.0", port=port)
