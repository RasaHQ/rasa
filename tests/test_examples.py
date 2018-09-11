from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import os
import sys

from rasa_core import training
from rasa_core.agent import Agent
from rasa_core.training import online
from rasa_core.events import ActionExecuted
from tests import utilities


def test_moodbot_example(trained_moodbot_path):
    agent = Agent.load(trained_moodbot_path)

    responses = agent.handle_text("/greet")
    assert responses[0]['text'] == 'Hey! How are you?'

    responses.extend(agent.handle_text("/mood_unhappy"))
    assert responses[-1]['text'] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 4


def test_restaurantbot_example():
    sys.path.append("examples/restaurantbot/")
    from bot import train_dialogue

    p = "examples/restaurantbot/"
    stories = os.path.join("data", "test_stories", "stories_babi_small.md")
    agent = train_dialogue(os.path.join(p, "restaurant_domain.yml"),
                           os.path.join(p, "models", "dialogue"),
                           stories)

    responses = agent.handle_text("/greet")
    assert responses[0]['text'] == 'how can I help you?'
