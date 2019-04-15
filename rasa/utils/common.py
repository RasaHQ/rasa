from typing import Callable, Text, List


def arguments_of(func: Callable) -> List[Text]:
    """Return the parameters of the function `func` as a list of names."""
    import inspect

    return list(inspect.signature(func).parameters.keys())
