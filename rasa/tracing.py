from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, cast

from jaeger_client import Config, Tracer
from jaeger_client.span import Span
from opentracing.ext import tags
from opentracing.propagation import Format
from opentracing.scope_managers.contextvars import ContextVarsScopeManager

tracer: Tracer
logger = logging.getLogger(__name__)


class IsAwaitable(Exception):
    ...


class IsNotAwaitable(Exception):
    ...


#             rasa.tracing.configure_tracing("rasa", "127.0.0.1", 6831, "b3", True)
# def configure_tracing(
#    app_name: str= 'rasa', host: str = '127.0.0.1', port: int = 831, propagation: str = b3, jaeger_enabled: bool = True
# ) -> Tracer:
def configure_tracing(tracing_endpoint: Any) -> Tracer:
    # tracing_endpoint is the endpoints.yml / tracing section
    # tracing:
    #   service_name: rasa
    #   agent_host: localhost
    #   agent_port: 6831
    #   propogation: b3
    global tracer
    host = "localhost"
    port = 6831
    propagation = "b3"
    service_name = "rasa"
    if tracing_endpoint:
        host = tracing_endpoint.kwargs.get("agent_host", "localhost")
        port = tracing_endpoint.kwargs.get("agent_port", 6831)
        propagation = tracing_endpoint.kwargs.get("propagation", "b3")
        service_name = tracing_endpoint.kwargs.get("service_name", "rasa")
    logger.info(f"initialize opentracing, agent_host: {host}, port: {port}")
    config = Config(
        config={
            "sampler": {"type": "const", "param": 1},
            "logging": False,
            "local_agent": {"reporting_host": host, "reporting_port": port},
            "propagation": propagation,
            "enabled": True,
        },
        validate=True,
        scope_manager=ContextVarsScopeManager(),
        service_name=service_name,
    )
    _tracer: Optional[Tracer] = config.initialize_tracer()
    if _tracer:
        tracer = _tracer
    return tracer


def get_tracer() -> Tracer:
    global tracer
    if not tracer:
        logger.warning("configure_tracing first")
    assert tracer
    return tracer


def get_current_span() -> Optional[Span]:
    tracer = get_tracer()
    active = tracer.scope_manager.active
    return cast(Span, active.span) if active else None


def get_current_trace_id() -> Optional[str]:
    span = get_current_span()
    trace_id = span.trace_id if span else None
    return "{:x}".format(trace_id) if trace_id else None


def get_tracing_http_headers() -> Dict[str, str]:
    tracer = get_tracer()
    span = get_current_span()
    if not span:
        return {}
    span.set_tag(tags.SPAN_KIND, tags.SPAN_KIND_RPC_CLIENT)
    headers: Dict[str, str] = {}
    tracer.inject(span, Format.HTTP_HEADERS, headers)
    return headers


def _get_span_and_scope(f: Callable[..., Any], *args: Any, **kwargs: Any):
    operation_name = f.__name__
    current_tags = {}
    if f.__qualname__ == "Agent.handle_message":
        if len(args) > 0 and args[0].input_channel:
            current_tags["channel"] = args[0].input_channel
            current_tags["message_id"] = args[0].message_id
            current_tags["sender_id"] = args[0].sender_id
            current_tags["text"] = args[0].text
    elif f.__qualname__ == "EndpointConfig.request":
        json = kwargs.get("json", None)
        if json and json["next_action"]:
            current_tags["next_action"] = json["next_action"]

    tracer = get_tracer()
    span = tracer.start_span(
        operation_name, child_of=get_current_span(), tags=current_tags
    )
    scope = tracer.scope_manager.activate(span, True)
    return span, scope


def trace_fn(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Trace synchronous functions

    Example
    ------
    >>> @trace_fn
    ... def any_function(*args, **kwargs):
    ...     pass
    ...
    """
    if inspect.iscoroutinefunction(f):
        raise IsAwaitable(
            f"{f.__qualname__} is awaitable. Use @trace_async_fn decorator instead."
        )

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        span, scope = _get_span_and_scope(f, **kwargs)
        try:
            result = f(*args, **kwargs)
        except Exception as error:
            span.set_tag(tags.ERROR, True)
            span.set_tag("error.message", str(error))
            raise error
        finally:
            scope.close()
        return result

    return wrapper


def trace_async_fn(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Trace async functions

    Example
    ------
    >>> @trace_async_fn
    ... async def any_function(*args, **kwargs):
    ...     pass
    ...
    """
    if not inspect.iscoroutinefunction(f):
        raise IsNotAwaitable(
            f"{f.__qualname__} is not awaitable. Use @trace_fn decorator instead."
        )

    @wraps(f)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        span, scope = _get_span_and_scope(f, *args, **kwargs)
        try:
            result = await f(*args, **kwargs)
        except Exception as error:
            span.set_tag(tags.ERROR, True)
            span.set_tag("error.message", str(error))
            raise error
        finally:
            scope.close()
        return result

    return wrapper


def trace_method(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Trace synchronous methods

    Example
    ------
    >>> class AnyClass:
    ...     @trace_method
    ...     def any_method(self, *args, **kwargs):
    ...         pass
    ...
    """
    if inspect.iscoroutinefunction(f):
        raise IsAwaitable(
            f"{f.__qualname__} is awaitable. Use @trace_async_method decorator instead."
        )

    @wraps(f)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        span, scope = _get_span_and_scope(f, *args, **kwargs)
        try:
            result = f(self, *args, **kwargs)
        except Exception as error:
            span.set_tag(tags.ERROR, True)
            span.set_tag("error.message", str(error))
            raise error
        finally:
            scope.close()
        return result

    return wrapper


def trace_async_method(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Trace asynchronus method

    Example
    ------
    >>> class AnyClass:
    ...     @trace_async_method
    ...     async def any_method(self, *args, **kwargs):
    ...         pass
    ...
    """
    if not inspect.iscoroutinefunction(f):
        raise IsNotAwaitable(
            f"{f.__qualname__} is not awaitable. Use @trace_method decorator instead."
        )

    @wraps(f)
    async def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        span, scope = _get_span_and_scope(f, *args, **kwargs)
        try:
            result = await f(self, *args, **kwargs)
        except Exception as error:
            span.set_tag(tags.ERROR, True)
            span.set_tag("error.message", str(error))
            raise error
        finally:
            scope.close()
        return result

    return wrapper


def trace_classmethod(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Trace synchronous class method

    Example
    ------
    >>> class AnyClass:
    ...     @classmethod
    ...     @trace_classmethod
    ...     def any_method(cls, *args, **kwargs):
    ...         pass
    ...
    """
    if inspect.iscoroutinefunction(f):
        raise IsAwaitable(
            f"{f.__qualname__} is awaitable. Use @trace_async_classmethod decorator instead."
        )

    @wraps(f)
    def wrapper(cls, *args: Any, **kwargs: Any) -> Any:
        span, scope = _get_span_and_scope(f, *args, **kwargs)
        try:
            result = f(cls, *args, **kwargs)
        except Exception as error:
            span.set_tag(tags.ERROR, True)
            span.set_tag("error.message", str(error))
            raise error
        finally:
            scope.close()
        return result

    return wrapper


def trace_async_classmethod(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Trace asynchronous class method

    Example
    ------
    >>> class AnyClass:
    ...     @classmethod
    ...     @trace_async_classmethod
    ...     async def any_method(cls, *args, **kwargs):
    ...         pass
    ...
    """
    if not inspect.iscoroutinefunction(f):
        raise IsNotAwaitable(
            f"{f.__qualname__} is not awaitable. Use @trace_classmethod decorator instead."
        )

    @wraps(f)
    async def wrapper(cls, *args: Any, **kwargs: Any) -> Any:
        span, scope = _get_span_and_scope(f, *args, **kwargs)
        try:
            result = await f(cls, *args, **kwargs)
        except Exception as error:
            span.set_tag(tags.ERROR, True)
            span.set_tag("error.message", str(error))
            raise error
        finally:
            scope.close()
        return result

    return wrapper


new_span = trace_async_classmethod
new_async_span = trace_async_classmethod
new_sync_span = trace_classmethod
