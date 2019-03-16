"""This module imports all of the components. To avoid cycles, no component
should import this in module scope."""

import logging
from typing import Any, Text

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
def policy_from_module_path(module_path: Text) -> Any:
    """Given the name of a policy module tries to retrieve the policy."""
    from rasa_core import utils
    from rasa_core.policies.keras_policy import KerasPolicy
    from rasa_core.policies.fallback import FallbackPolicy
    from rasa_core.policies.two_stage_fallback import TwoStageFallbackPolicy
    from rasa_core.policies.memoization import (
        MemoizationPolicy,
        AugmentedMemoizationPolicy)
    from rasa_core.policies.embedding_policy import EmbeddingPolicy
    from rasa_core.policies.form_policy import FormPolicy
    from rasa_core.policies.sklearn_policy import SklearnPolicy

    try:
        return utils.class_from_module_path(module_path)
    except ImportError:
        raise ImportError("Cannot retrieve policy from path '{}'"
                          "".format(module_path))


# noinspection PyUnresolvedReferences
def featurizer_from_module_path(module_path: Text) -> Any:
    """Given the name of a featurizer module tries to retrieve it."""
    from rasa_core import utils
    from rasa_core.featurizers import (
        FullDialogueTrackerFeaturizer,
        MaxHistoryTrackerFeaturizer,
        BinarySingleStateFeaturizer,
        LabelTokenizerSingleStateFeaturizer)

    try:
        return utils.class_from_module_path(module_path)
    except ImportError:
        raise ImportError("Cannot retrieve featurizer from path '{}'"
                          "".format(module_path))
