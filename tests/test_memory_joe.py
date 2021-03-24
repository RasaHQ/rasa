from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter

from rasa.core import training
import rasa.utils.common

default_domain = Domain.load("data/test_domains/default_with_slots.yml")
training_trackers = rasa.utils.common.run_in_loop(
    training.load_data(
        "data/test_yaml_stories/stories_defaultdomain.yml",
        default_domain,
        augmentation_factor=20,
    )
)

featurizer = MaxHistoryTrackerFeaturizer(SingleStateFeaturizer(), max_history=None)
policy = TEDPolicy(featurizer=featurizer, epochs=500)

policy.train(training_trackers, default_domain, RegexInterpreter())

"""
Run:
    mprof run python tests/test_memory_joe.py
    mprof plot
"""
