from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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


def test_concerts_online_example(tmpdir):
    sys.path.append("examples/concertbot/")
    from train_online import train_agent
    from rasa_core import utils

    story_path = tmpdir.join("stories.md").strpath

    with utilities.cwd("examples/concertbot"):
        msgs = iter(["/greet", "/greet", "/greet"])

        with utilities.mocked_cmd_input(
                utils,
                text=["2",  # action is wrong
                      "5",  # choose utter_goodbye action
                      "1"  # yes, action_listen is correct.
                      ] * 2 + [  # repeat this twice
                         "0",  # export
                         story_path  # file path to export to
                     ]):
            agent = train_agent()

            responses = agent.handle_text("/greet", sender_id="user1")
            assert responses[-1]['text'] == "hey there!"

            online.serve_agent(agent, get_next_message=msgs.next)

            # the model should have been retrained and the model should now
            # directly respond with goodbye
            responses = agent.handle_text("/greet", sender_id="user2")
            assert responses[-1]['text'] == "goodbye :("

            assert os.path.exists(story_path)
            print(utils.read_file(story_path))

            t = training.load_data(story_path, agent.domain,
                                   use_story_concatenation=False)
            assert len(t) == 1
            assert len(t[0].events) == 9
            assert t[0].events[5] == ActionExecuted("utter_goodbye")
            assert t[0].events[6] == ActionExecuted("action_listen")
