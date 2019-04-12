from typing import Callable


def arguments_of(func: Callable):
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())
