import copy

from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.interpreter import RegexInterpreter
from tests.core.conftest import DEFAULT_DOMAIN_PATH_WITH_SLOTS, DEFAULT_STORIES_FILE
from rasa.core import training
import rasa.utils.common
import tensorflow as tf
import gc
from pympler import muppy, summary

def print_total_num_objects():
    all_objects = muppy.get_objects()
    print(len(all_objects))

def print_mem_summary():
    sum1 = summary.summarize(muppy.get_objects())
    summary.print_(sum1)


default_domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
training_trackers = rasa.utils.common.run_in_loop(
    training.load_data(DEFAULT_STORIES_FILE, default_domain, augmentation_factor=20)
)
featurizer = MaxHistoryTrackerFeaturizer(SingleStateFeaturizer(), max_history=None)
policy = TEDPolicy(featurizer=featurizer, epochs=500)
policy.train(training_trackers, default_domain, RegexInterpreter())


"""
Run:
    mprof run python leak_test.py
    mprof plot
"""


