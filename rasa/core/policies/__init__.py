# we need to import the policy first
from rasa.core.policies.policy import Policy

# and after that any implementation
from rasa.core.policies.ensemble import SimplePolicyEnsemble, PolicyEnsemble
