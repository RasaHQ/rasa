from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.featurizers import BinaryFeaturizer
from rasa_core.policies.ensemble import PolicyEnsemble


def test_domain_spec_dm():
    model_path = 'examples/babi/models/policy/current'
    policy = PolicyEnsemble.load(model_path, BinaryFeaturizer())
    policy.persist(model_path)
