# Import all policies at one place to be able to to resolve them via a common module
# path. Don't do this in `__init__.py` to avoid importing them without need.

from rasa.core.policies.ted_policy import TEDPolicy  # noqa: F401

from rasa.core.policies.fallback import FallbackPolicy  # noqa: F401

from rasa.core.policies.memoization import (  # noqa: F401
    MemoizationPolicy,
    AugmentedMemoizationPolicy,
)

from rasa.core.policies.sklearn_policy import SklearnPolicy  # noqa: F401

from rasa.core.policies.form_policy import FormPolicy  # noqa: F401

from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy  # noqa: F401

from rasa.core.policies.mapping_policy import MappingPolicy  # noqa: F401

from rasa.core.policies.rule_policy import RulePolicy  # noqa: F401

from rasa.core.policies.unexpected_intent_policy import (  # noqa: F401
    UnexpecTEDIntentPolicy,
)
