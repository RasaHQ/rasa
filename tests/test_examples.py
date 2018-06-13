from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

from rasa_core.channels.file import FileInputChannel
from rasa_core.interpreter import RegexInterpreter


def test_moodbot_example(trained_moodbot_path):
    from rasa_core import run

    agent = run.main(trained_moodbot_path)

    responses = agent.handle_message("/greet")
    assert responses[0]['text'] == 'Hey! How are you?'

    responses.extend(agent.handle_message("/mood_unhappy"))
    assert responses[-1]['text'] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 4


def test_remote_example():
    from rasa_core import train, run
    from rasa_core.events import SlotSet

    train.train_dialogue_model(
            domain_file="examples/remotebot/concert_domain_remote.yml",
            stories_file="examples/remotebot/data/stories.md",
            output_path="examples/remotebot/models/dialogue",
            use_online_learning=False,
            nlu_model_path=None,
            max_history=None,
            kwargs=None
    )
    agent = run.main("examples/remotebot/models/dialogue")

    response = agent.start_message_handling("/search_venues")
    assert response.get("next_action") == 'search_venues'

    reference = {
        'slots': {'concerts': None, 'venues': None},
        'events': None,
        'sender_id': 'default',
        'paused': False,
        'latest_event_time': 1513023382.101372,
        'latest_message': {
            'text': '/search_venues',
            'intent_ranking': [{'confidence': 1.0, 'name': 'search_venues'}],
            'intent': {'confidence': 1.0, 'name': 'search_venues'},
            'entities': []}}
    result = response.get("tracker")

    assert reference.keys() == result.keys()
    del reference['latest_event_time']
    del result['latest_event_time']
    assert reference == result

    venues = [{"name": "Big Arena", "reviews": 4.5}]
    next_response = agent.continue_message_handling(
        "default", "search_venues", [SlotSet("venues", venues)]
    )
    assert next_response.get("next_action") == "action_listen"


def test_restaurantbot_example():
    sys.path.append("examples/restaurantbot/")
    from bot import train_dialogue

    p = "examples/restaurantbot/"
    stories = os.path.join("data", "test_stories", "stories_babi_small.md")
    agent = train_dialogue(os.path.join(p, "restaurant_domain.yml"),
                           os.path.join(p, "models", "dialogue"),
                           stories)

    responses = agent.handle_message("/greet")
    assert responses[0]['text'] == 'how can I help you?'


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
    responses = agent.handle_message("/greet")
    assert responses[-1]['text'] in {"hey there!",
                                     "how can I help you?",
                                     "default message"}
