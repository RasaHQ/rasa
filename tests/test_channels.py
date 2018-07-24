from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels.console import ConsoleInputChannel

def test_console_input():
    import rasa_core.channels.console
    # Overwrites the input() function and when someone else tries to read
    # something from the command line this function gets called. But instead of
    # waiting input for the user, this simulates the input of
    # "2", therefore it looks like the user is always typing "2" if someone
    # requests a cmd input.

    rasa_core.channels.console.input = lambda _=None: "Test Input"

    recorded = []

    def on_message(message):
        recorded.append(message)
    channel = ConsoleInputChannel()
    channel._record_messages(on_message, max_message_limit=3)
    assert [r.text for r in recorded] == ["Test Input",
                                          "Test Input",
                                          "Test Input"]

def test_slack_init_one_parameter():
    import rasa_core.channels.slack
    ch = rasa_core.channels.slack.SlackInput("xoxb-test")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == None

def test_slack_init_two_parameters():
    import rasa_core.channels.slack
    ch = rasa_core.channels.slack.SlackInput("xoxb-test","test")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == "test"

def test_is_slack_message_none():
    import rasa_core.channels.slack
    import json
    payload = {}
    slack_message = json.loads(json.dumps(payload))
    assert rasa_core.channels.slack.SlackInput._is_user_message(slack_message) == None

def test_is_slack_message_true():
    import rasa_core.channels.slack
    import json
    event = {}
    event['type'] = 'message'
    event['channel'] = 'C2147483705'
    event['user'] = 'U2147483697'
    event['text'] = 'Hello world'
    event['ts'] = '1355517523'
    payload = json.dumps({'event':event}) 
    slack_message = json.loads(payload)
    assert rasa_core.channels.slack.SlackInput._is_user_message(slack_message) == True

def test_is_slack_message_false():
    import rasa_core.channels.slack
    import json
    event = {}
    event['type'] = 'message'
    event['channel'] = 'C2147483705'
    event['user'] = 'U2147483697'
    event['text'] = 'Hello world'
    event['ts'] = '1355517523'
    event['bot_id'] = '1355517523' # This should result in false, even if everything were true.
    payload = json.dumps({'event':event}) 
    slack_message = json.loads(payload)
    assert rasa_core.channels.slack.SlackInput._is_user_message(slack_message) == False
