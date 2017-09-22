from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.domain import TemplateDomain
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.policies import Policy
from rasa_core.tracker_store import InMemoryTrackerStore

logger = logging.getLogger(__name__)


class SimplePolicy(Policy):
    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        responses = {
            "greet": 3,
        }

        if tracker.latest_action_name == ACTION_LISTEN_NAME:
            key = tracker.latest_message.intent["name"]
            action = responses[key] if key in responses else 2
            return utils.one_hot(action, domain.num_actions)
        else:
            return np.zeros(domain.num_actions)


class HelloInterpreter(NaturalLanguageInterpreter):
    def parse(self, message):
        intent = "greet" if 'hello' in message else "default"
        return {
            "text": message,
            "intent": {"name": intent, "confidence": 1.0},
            "entities": []
        }


def run_hello_world(serve_forever=True):
    default_domain = TemplateDomain.load("examples/default_domain.yml")
    agent = Agent(default_domain,
                  policies=[SimplePolicy()],
                  interpreter=HelloInterpreter(),
                  tracker_store=InMemoryTrackerStore(default_domain))

    if serve_forever:
        # Attach the commandline input to the controller to handle all
        # incoming messages from that channel
        agent.handle_channel(ConsoleInputChannel())

    return agent


if __name__ == '__main__':
    run_hello_world()
