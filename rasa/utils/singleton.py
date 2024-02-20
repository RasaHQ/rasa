from typing import Any, Dict


class Singleton(type):
    """Singleton metaclass."""

    _instances: Dict[Any, Any] = {}  # noqa: RUF012

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Call the class.

        Args:
            *args: Arguments.
            **kwargs: Keyword arguments.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]

    def clear(cls) -> None:
        """Clear the class."""
        cls._instances = {}
