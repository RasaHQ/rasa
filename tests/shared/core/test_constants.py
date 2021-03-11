import pytest

from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.policies.rule_policy import RulePolicy
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from rasa.shared.core.constants import (
    POLICY_NAME_FALLBACK,
    POLICY_NAME_MAPPING,
    POLICY_NAME_RULE,
    POLICY_NAME_TWO_STAGE_FALLBACK,
    POLICY_NAME_FORM,
)


@pytest.mark.parametrize(
    "name_in_constant, policy_class",
    [
        (POLICY_NAME_TWO_STAGE_FALLBACK, TwoStageFallbackPolicy),
        (POLICY_NAME_FALLBACK, FallbackPolicy),
        (POLICY_NAME_MAPPING, MappingPolicy),
        (POLICY_NAME_FORM, FormPolicy),
        (POLICY_NAME_RULE, RulePolicy),
    ],
)
def test_policy_names(name_in_constant: Text, policy_class: Type):
    assert name_in_constant == policy_class.__name__
