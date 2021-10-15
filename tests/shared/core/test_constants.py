from typing import Text, Type

import pytest

from rasa.core.policies.rule_policy import RulePolicyGraphComponent
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifierGraphComponent
from rasa.shared.core.constants import (
    CLASSIFIER_NAME_FALLBACK,
    POLICY_NAME_RULE,
)


@pytest.mark.parametrize(
    "name_in_constant, policy_class",
    [
        (POLICY_NAME_RULE, RulePolicyGraphComponent),
        (CLASSIFIER_NAME_FALLBACK, FallbackClassifierGraphComponent),
    ],
)
def test_policy_names(name_in_constant: Text, policy_class: Type):
    # If this raises it means that we have removed the `GraphComponent` suffix as part
    # of the architecture revamp. It's safe to remove the following line and to to drop
    # the `replace("GraphComponent", "")` part
    assert policy_class.__name__.endswith("GraphComponent")
    assert name_in_constant == policy_class.__name__.replace("GraphComponent", "")


@pytest.mark.parametrize(
    "name_in_constant, classifier_class",
    [(CLASSIFIER_NAME_FALLBACK, FallbackClassifierGraphComponent),],
)
def test_classifier_names(name_in_constant: Text, classifier_class: Type):
    # If this raises it means that we have removed the `GraphComponent` suffix as part
    # of the architecture revamp. It's safe to remove the following line and to to drop
    # the `replace("GraphComponent", "")` part
    assert classifier_class.__name__.endswith("GraphComponent")
    assert name_in_constant == classifier_class.__name__.replace("GraphComponent", "")
