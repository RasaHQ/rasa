"""This module imports all of the components. To avoid cycles, no component
should import this in module scope."""

import logging
import typing
from typing import Text, Type

if typing.TYPE_CHECKING:
    from rasa.core.policies.policy import Policy
    from rasa.core.featurizers import TrackerFeaturizer

logger = logging.getLogger(__name__)


def policy_from_module_path(module_path: Text) -> Type["Policy"]:
    """Given the name of a policy module tries to retrieve the policy."""
    from rasa.utils.common import class_from_module_path

    try:
        return class_from_module_path(
            module_path, lookup_path="rasa.core.policies.registry"
        )
    except ImportError:
        raise ImportError(f"Cannot retrieve policy from path '{module_path}'")


def featurizer_from_module_path(module_path: Text) -> Type["TrackerFeaturizer"]:
    """Given the name of a featurizer module tries to retrieve it."""
    from rasa.utils.common import class_from_module_path

    try:
        return class_from_module_path(module_path, lookup_path="rasa.core.featurizers")
    except ImportError:
        raise ImportError(f"Cannot retrieve featurizer from path '{module_path}'")
