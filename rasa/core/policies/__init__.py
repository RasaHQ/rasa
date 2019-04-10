# we need to import the policy first
from rasa.core.policies.policy import Policy

pass
# and after that any implementation
from rasa.core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
from rasa.core.policies.embedding_policy import EmbeddingPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.keras_policy import KerasPolicy
from rasa.core.policies.memoization import MemoizationPolicy, AugmentedMemoizationPolicy
from rasa.core.policies.sklearn_policy import SklearnPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
