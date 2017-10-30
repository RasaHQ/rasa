from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels import UserMessage
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.featurizers import BinaryFeaturizer
from rasa_core.interpreter import RegexInterpreter
from rasa_core.channels.console import ConsoleOutputChannel
from rasa_core.policies import PolicyTrainer
from rasa_core.policies.ensemble import SimplePolicyEnsemble
from rasa_core.policies.scoring_policy import ScoringPolicy
from rasa_core.processor import MessageProcessor
from rasa_core.tracker_store import InMemoryTrackerStore


def test_message_processor(default_domain, capsys):
    story_filename = "data/dsl_stories/stories_defaultdomain.md"
    ensemble = SimplePolicyEnsemble([ScoringPolicy()])
    interpreter = RegexInterpreter()

    PolicyTrainer(ensemble, default_domain, BinaryFeaturizer()).train(
            story_filename,
            max_history=3)

    tracker_store = InMemoryTrackerStore(default_domain)
    processor = MessageProcessor(interpreter,
                                 ensemble,
                                 default_domain,
                                 tracker_store)

    out = CollectingOutputChannel()
    processor.handle_message(UserMessage("_greet[name=Core]", out))
    assert ("default", "hey there Core!") == out.latest_output()
