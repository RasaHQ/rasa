# we need to import the policy first
from rasa_core.policies.policy import Policy

pass
# and after that any implementation
from rasa_core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa_core.policies.embedding_policy import EmbeddingPolicy
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import (
    MemoizationPolicy, AugmentedMemoizationPolicy)
from rasa_core.policies.sklearn_policy import SklearnPolicy
from rasa_core.policies.form_policy import FormPolicy
from rasa_core.policies.two_stage_fallback import TwoStageFallbackPolicy
