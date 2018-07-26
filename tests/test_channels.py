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

def test_slackinput_init_one_parameter():
    import rasa_core.channels.slack
    ch = rasa_core.channels.slack.SlackInput("xoxb-test")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == None

def test_slackinput_init_two_parameters():
    import rasa_core.channels.slack
    ch = rasa_core.channels.slack.SlackInput("xoxb-test","test")
    assert ch.slack_token == "xoxb-test"
    assert ch.slack_channel == "test"

def test_is_slackinput_message_none():
    import rasa_core.channels.slack
    import json
    payload = {}
    slack_message = json.loads(json.dumps(payload))
    assert rasa_core.channels.slack.SlackInput._is_user_message(slack_message) == None

def test_is_slackinput_message_true():
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

def test_is_slackinput_message_false():
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

def test_slackbot_init_one_parameter():
    import rasa_core.channels.slack
    ch = rasa_core.channels.slack.SlackBot("DummyToken")
    assert ch.token == "DummyToken"
    assert ch.slack_channel == None

def test_slackbot_init_two_parameter():
    import rasa_core.channels.slack
    bot = rasa_core.channels.slack.SlackBot("DummyToken","General")
    assert bot.token == "DummyToken"
    assert bot.slack_channel == "General"

#Use monkeypatch for sending attachments, images and plain text.
def test_slackbot_send_attachment_only(monkeypatch):
    def mockreturn(self,method,channel,as_user,text,attachments):
        return attachments
    import rasa_core.channels.slack
    import slackclient 
    import json
    monkeypatch.setattr(slackclient.SlackClient, 'api_call', mockreturn)
    bot = rasa_core.channels.slack.SlackBot("DummyToken","General")
    attachment=json.dumps([{"fallback": "Financial Advisor Summary", "color": "#36a64f", "author_name": "TABE", "title": "Financial Advisor Summary", "title_link": "http://tenfactorialrocks.com", "image_url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "thumb_url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "actions": [{"type": "button", "text": "\ud83d\udcc8 Go to Dashboard", "url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "style": "primary"}, {"type": "button", "text": "\ud83d\udccb Download XL", "url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "style": "danger"}, {"type": "button", "text": "\ud83d\udce7 E-Mail", "url": "https://requests.example.com/cancel/r123456", "style": "danger"}], "footer": "Powered by 1010rocks", "ts": 1531889719.0970402}])
    assert bot.send_attachment("ID", attachment) == attachment

def test_slackbot_send_attachment_withtext(monkeypatch):
    def mockreturn(self,method,channel,as_user,text,attachments):
        return attachments+text
    import rasa_core.channels.slack
    import slackclient 
    import json
    monkeypatch.setattr(slackclient.SlackClient, 'api_call', mockreturn)
    bot = rasa_core.channels.slack.SlackBot("DummyToken","General")
    text="Sample text"
    attachment=json.dumps([{"fallback": "Financial Advisor Summary", "color": "#36a64f", "author_name": "TABE", "title": "Financial Advisor Summary", "title_link": "http://tenfactorialrocks.com", "image_url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "thumb_url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "actions": [{"type": "button", "text": "\ud83d\udcc8 Go to Dashboard", "url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "style": "primary"}, {"type": "button", "text": "\ud83d\udccb Download XL", "url": "http://tenfactorialrocks.com/wp-content/uploads/2018/02/3-wise-summary.png", "style": "danger"}, {"type": "button", "text": "\ud83d\udce7 E-Mail", "url": "https://requests.example.com/cancel/r123456", "style": "danger"}], "footer": "Powered by 1010rocks", "ts": 1531889719.0970402}])
    assert bot.send_attachment("ID", attachment,text) == attachment+text

def test_slackbot_send_image_url(monkeypatch):
    def mockreturn(self,method,channel,as_user,attachments):
        return json.dumps(attachments)
    import rasa_core.channels.slack
    import slackclient 
    import json
    monkeypatch.setattr(slackclient.SlackClient, 'api_call', mockreturn)
    bot = rasa_core.channels.slack.SlackBot("DummyToken","General")
    url=json.dumps([{"URL": "http://www.rasa.net"}])
    assert bot.send_image_url("ID", url) == json.dumps([{'image_url': '[{"URL": "http://www.rasa.net"}]', 'text': ''}])

def test_slackbot_send_text(monkeypatch):
    def mockreturn(self,method,channel,as_user,text):
        return text
    import rasa_core.channels.slack
    import slackclient 
    monkeypatch.setattr(slackclient.SlackClient, 'api_call', mockreturn)
    bot = rasa_core.channels.slack.SlackBot("DummyToken","General")
    text = "Some text" # This text is returned back by the mock.
    assert bot.send_text_message("ID", text) == text
