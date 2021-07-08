# we need to import the policy first
from rasa.core.policies.policy import Policy  # noqa: F401

# and after that any implementation
from rasa.core.policies.ensemble import (  # noqa: F401
    SimplePolicyEnsemble,
    PolicyEnsemble,
)
