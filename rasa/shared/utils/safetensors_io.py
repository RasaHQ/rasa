"""Lazy access to safetensors.numpy I/O with a single error message."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

_DEFAULT_MISSING_MESSAGE = (
    "This operation requires safetensors. Install the NLU extra: "
    "pip install 'rasa-pro[nlu]' OR poetry add 'rasa-pro[nlu]'"
)


def safetensors_numpy_load_save(
    *,
    missing_dependency_message: Optional[str] = None,
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Return ``(load_file, save_file)`` from ``safetensors.numpy``.

    Args:
        missing_dependency_message: If safetensors is not installed, raise
            ``MissingDependencyException`` with this text; otherwise use the
            default NLU install hint.

    Returns:
        The ``load_file`` and ``save_file`` callables from ``safetensors.numpy``.

    Raises:
        MissingDependencyException: When ``safetensors.numpy`` cannot be imported.
    """
    try:
        from safetensors.numpy import load_file, save_file
    except ImportError:
        from rasa.exceptions import MissingDependencyException

        raise MissingDependencyException(
            missing_dependency_message or _DEFAULT_MISSING_MESSAGE
        ) from None
    return load_file, save_file
