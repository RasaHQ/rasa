from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json

from httpretty import httpretty

from rasa_core import utils
from rasa_core.channels import console
from tests import utilities
from tests.conftest import MOODBOT_MODEL_PATH

# this is needed so that the tests included as code examples look better
MODEL_PATH = MOODBOT_MODEL_PATH


def test_console_input():
    import rasa_core.channels.console

    # Overwrites the input() function and when someone else tries to read
    # something from the command line this function gets called.
    with utilities.mocked_cmd_input(rasa_core.channels.console,
                                    text="Test Input"):
        httpretty.register_uri(httpretty.POST,
                               'https://abc.defg/webhooks/rest/webhook',
                               body='')

        httpretty.enable()
        console.record_messages(
                server_url="https://abc.defg",
                max_message_limit=3)
        httpretty.disable()

        assert (httpretty.latest_requests[-1].path ==
                "/webhooks/rest/webhook?stream=true&token=")

        b = httpretty.latest_requests[-1].body.decode("utf-8")

        assert json.loads(b) == {"message": "Test Input", "sender": "default"}


# USED FOR DOCS - don't rename without changing in the docs
def test_facebook_channel():
    from rasa_core.channels.facebook import FacebookInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load(MODEL_PATH, interpreter=RegexInterpreter())

    input_channel = FacebookInput(
            fb_verify="YOUR_FB_VERIFY",
            # you need tell facebook this token, to confirm your URL
            fb_secret="YOUR_FB_SECRET",  # your app secret
            fb_access_token="YOUR_FB_PAGE_ACCESS_TOKEN"
            # token for the page you subscribed to
    )

    # set serve_forever=False if you want to keep the server running
    s = agent.handle_channels([input_channel], 5004, serve_forever=False)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    assert s.started
    routes_list = utils.list_routes(s.application)
    assert routes_list.get("/webhooks/facebook/").startswith(
            'fb_webhook.health')
    assert routes_list.get("/webhooks/facebook/webhook").startswith(
            'fb_webhook.webhook')
    s.stop()


# USED FOR DOCS - don't rename without changing in the docs
def test_slack_channel():
    from rasa_core.channels.slack import SlackInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load(MODEL_PATH, interpreter=RegexInterpreter())

    input_channel = SlackInput(
            slack_token="YOUR_SLACK_TOKEN",
            # this is the `bot_user_o_auth_access_token`
            slack_channel="YOUR_SLACK_CHANNEL"
            # the name of your channel to which the bot posts (optional)
    )

    # set serve_forever=False if you want to keep the server running
    s = agent.handle_channels([input_channel], 5004, serve_forever=False)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    assert s.started
    routes_list = utils.list_routes(s.application)
    assert routes_list.get("/webhooks/slack/").startswith(
            'slack_webhook.health')
    assert routes_list.get("/webhooks/slack/webhook").startswith(
            'slack_webhook.webhook')
    s.stop()


# USED FOR DOCS - don't rename without changing in the docs
def test_mattermost_channel():
    from rasa_core.channels.mattermost import MattermostInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load(MODEL_PATH, interpreter=RegexInterpreter())

    input_channel = MattermostInput(
            # this is the url of the api for your mattermost instance
            url="http://chat.example.com/api/v4",
            # the name of your team for mattermost
            team="community",
            # the username of your bot user that will post
            user="user@email.com",
            # messages
            pw="password"
            # the password of your bot user that will post messages
    )

    # set serve_forever=False if you want to keep the server running
    s = agent.handle_channels([input_channel], 5004, serve_forever=False)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    assert s.started
    routes_list = utils.list_routes(s.application)
    assert routes_list.get("/webhooks/mattermost/").startswith(
            'mattermost_webhook.health')
    assert routes_list.get("/webhooks/mattermost/webhook").startswith(
            'mattermost_webhook.webhook')
    s.stop()


# USED FOR DOCS - don't rename without changing in the docs
def test_telegram_channel():
    # telegram channel will try to set a webhook, so we need to mock the api

    httpretty.register_uri(
            httpretty.POST,
            'https://api.telegram.org/bot123:YOUR_ACCESS_TOKEN/setWebhook',
            body='{"ok": true, "result": {}}')

    httpretty.enable()

    from rasa_core.channels.telegram import TelegramInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load(MODEL_PATH, interpreter=RegexInterpreter())

    input_channel = TelegramInput(
            # you get this when setting up a bot
            access_token="123:YOUR_ACCESS_TOKEN",
            # this is your bots username
            verify="YOUR_TELEGRAM_BOT",
            # the url your bot should listen for messages
            webhook_url="YOUR_WEBHOOK_URL"
    )

    # set serve_forever=False if you want to keep the server running
    s = agent.handle_channels([input_channel], 5004, serve_forever=False)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    assert s.started
    routes_list = utils.list_routes(s.application)
    assert routes_list.get("/webhooks/telegram/").startswith(
            'telegram_webhook.health')
    assert routes_list.get("/webhooks/telegram/webhook").startswith(
            'telegram_webhook.message')
    s.stop()
    httpretty.disable()


# USED FOR DOCS - don't rename without changing in the docs
def test_twilio_channel():
    from rasa_core.channels.twilio import TwilioInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load(MODEL_PATH, interpreter=RegexInterpreter())

    input_channel = TwilioInput(
            # you get this from your twilio account
            account_sid="YOUR_ACCOUNT_SID",
            # also from your twilio account
            auth_token="YOUR_AUTH_TOKEN",
            # a number associated with your twilio account
            twilio_number="YOUR_TWILIO_NUMBER"
    )

    # set serve_forever=False if you want to keep the server running
    s = agent.handle_channels([input_channel], 5004, serve_forever=False)
    # END DOC INCLUDE
    # the above marker marks the end of the code snipped included
    # in the docs
    assert s.started
    routes_list = utils.list_routes(s.application)
    assert routes_list.get("/webhooks/twilio/").startswith(
            'twilio_webhook.health')
    assert routes_list.get("/webhooks/twilio/webhook").startswith(
            'twilio_webhook.message')
    s.stop()
