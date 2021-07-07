import logging

from typing import Callable, Any, Dict

import rasa.shared.utils.common

logger = logging.getLogger(__name__)


def filter_matching_kwargs_and_warn(func: Callable, **kwargs: Any) -> Dict:
    r"""Filters out kwargs that are not explicitly named in func's signature.

    Note that keyword arguments that func might accept via **kwargs will be filtered
    out by this function (unless they are passed as "kwargs = {...}" to this function)
    and a warning will be raised.

    Args:
        func: a callable function
        **kwargs: keyword arguments supposed to be passed on to func
    Returns:
        a dictionary of keyword arguments where each key matches one parameter name
        of func's signature
    """
    valid_keys = rasa.shared.utils.common.arguments_of(func)

    # NOTE: before this would pass on "invalid" kwargs if their value evaluated to False...
    params = {key: kwargs[key] for key in valid_keys if key in kwargs}
    ignored_params = {key: val for key, val in kwargs.items() if key not in valid_keys}

    # TODO: this should be a warning
    logger.debug(f"Parameters ignored by `model.fit(...)`: {ignored_params}")

    return params
