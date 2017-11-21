from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import os

from rasa_core.channels.file import FileInputChannel
from rasa_core.interpreter import RegexInterpreter


def test_moodbot_example():
    from rasa_core import train, run

    train.train_dialogue_model("examples/moodbot/domain.yml",
                               "examples/moodbot/data/stories.md",
                               "examples/moodbot/models/dialogue",
                               False, None, {})
    agent = run.main("examples/moodbot/models/dialogue")

    responses = agent.handle_message("_greet")
    assert responses[0] == 'Hey! How are you?'

    responses.extend(agent.handle_message("_mood_unhappy"))
    assert responses[-1] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 6


def test_remote_example():
    from rasa_core import train, run

    train.train_dialogue_model("examples/remotebot/concert_domain_remote.yml",
                               "examples/remotebot/data/stories.md",
                               "examples/remotebot/models/dialogue",
                               False, None, {})
    agent = run.main("examples/remotebot/models/dialogue")

    response = agent.start_message_handling("_search_venues")
    assert response.get("next_action") == 'search_venues'
    assert response.get("tracker") == {
        'slots': {'concerts': None, 'venues': None},
        'sender_id': 'default',
        'latest_message': {
            'text': '_search_venues',
            'intent_ranking': [{'confidence': 1.0, 'name': 'search_venues'}],
            'intent': {'confidence': 1.0, 'name': 'search_venues'},
            'entities': []}}

    next_response = agent.continue_message_handling("default", "search_venues",
                                                    [])
    assert next_response.get("next_action") == "action_listen"


def test_restaurantbot_example():
    sys.path.append("examples/restaurantbot/")
    from bot import train_dialogue

    p = "examples/restaurantbot/"
    agent = train_dialogue(os.path.join(p, "restaurant_domain.yml"),
                           os.path.join(p, "models", "dialogue"),
                           os.path.join(p, "data", "babi_stories.md"))

    responses = agent.handle_message("_greet")
    assert responses[0] == 'how can I help you?'


def test_concerts_online_example():
    sys.path.append("examples/concertbot/")
    from train_online import run_concertbot_online
    from rasa_core import utils

    # simulates cmdline input / detailed explanation above
    utils.input = lambda _=None: "2"

    input_channel = FileInputChannel(
            'examples/concertbot/data/stories.md',
            message_line_pattern='^\s*\*\s(.*)$',
            max_messages=3)
    domain_file = os.path.join("examples", "concertbot", "concert_domain.yml")
    training_file = os.path.join("examples", "concertbot", "data", "stories.md")
    agent = run_concertbot_online(input_channel, RegexInterpreter(),
                                  domain_file, training_file)
    responses = agent.handle_message("_greet")
    assert responses[-1] in {"hey there!",
                             "how can I help you?",
                             "default message"}
