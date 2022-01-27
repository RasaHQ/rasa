from typing import Text, Type

import pytest

from rasa.core.policies.rule_policy import RulePolicy
from rasa.nlu.classifiers.fallback_classifier import FallbackClassifier
from rasa.shared.core.constants import CLASSIFIER_NAME_FALLBACK, POLICY_NAME_RULE


@pytest.mark.parametrize(
    "name_in_constant, policy_class",
    [(POLICY_NAME_RULE, RulePolicy), (CLASSIFIER_NAME_FALLBACK, FallbackClassifier)],
)
def test_policy_names(name_in_constant: Text, policy_class: Type):
    assert name_in_constant == policy_class.__name__


@pytest.mark.parametrize(
    "name_in_constant, classifier_class",
    [(CLASSIFIER_NAME_FALLBACK, FallbackClassifier)],
)
def test_classifier_names(name_in_constant: Text, classifier_class: Type):
    assert name_in_constant == classifier_class.__name__
