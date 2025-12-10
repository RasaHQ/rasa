"""Centralized langfuse import handling.

All langfuse imports MUST go through this module. Direct imports of langfuse
are banned via ruff (see pyproject.toml banned-api configuration).

This provides:
- Graceful fallback when langfuse is not installed (optional dependency)
- No-op implementations for decorators like @observe
- Context manager that only executes block when langfuse is available

Usage:
    from rasa.builder.telemetry.langfuse_compat import observe, with_langfuse

    # For decorators (works whether langfuse is installed or not):
    @observe()
    def my_function(): ...

    # For conditional code that requires langfuse:
    with with_langfuse() as lf:
        if lf:  # Block only runs if langfuse is available
            client = lf.get_client()
"""

from contextlib import contextmanager
from functools import wraps
from types import ModuleType
from typing import Any, Callable, Generator, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _no_op_observe(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """No-op implementation of langfuse.observe decorator that preserves types."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*f_args: Any, **f_kwargs: Any) -> Any:
            return func(*f_args, **f_kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


try:
    import langfuse as _langfuse  # noqa: TID251

    _LANGFUSE_AVAILABLE = True
    langfuse: Any = _langfuse
except ImportError:
    _LANGFUSE_AVAILABLE = False

    class _MockLangfuse:
        """Mock langfuse module when the real one isn't installed."""

        observe = staticmethod(_no_op_observe)

        def get_client(self) -> None:
            return None

    langfuse = _MockLangfuse()


@contextmanager
def with_langfuse() -> Generator[Optional[ModuleType], None, None]:
    """Context manager that yields the langfuse module if available.

    If langfuse is not installed, yields None and the block still executes
    but the caller must check the yielded value.

    Usage:
        with with_langfuse() as lf:
            if lf:
                lf.get_client().update_current_trace(...)
    """
    if _LANGFUSE_AVAILABLE:
        yield _langfuse
    else:
        yield None


def is_langfuse_available() -> bool:
    """Check if langfuse is available.

    Use this for decoration-time checks where context manager isn't suitable.
    """
    return _LANGFUSE_AVAILABLE


def require_langfuse() -> ModuleType:
    """Return the langfuse module or raise ImportError if not available.

    Use this in modules that absolutely require langfuse (e.g., evaluators).

    Raises:
        ImportError: If langfuse is not installed.
    """
    if not _LANGFUSE_AVAILABLE:
        raise ImportError(
            "langfuse is required for this functionality. "
            "Install it with: pip install rasa-pro[monitoring]"
        )
    return _langfuse


# Re-export for convenience - use the typed no-op when langfuse unavailable
def observe(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed observe decorator that works whether langfuse is available or not."""
    return langfuse.observe(*args, **kwargs)
