from rasa.core import run

CREDENTIALS_FILE = "examples/moodbot/credentials.yml"


def test_create_http_input_channels():
    channels = run.create_http_input_channels(None, CREDENTIALS_FILE)
    assert len(channels) == 7

    # ensure correct order
    assert {c.name() for c in channels} == {
        "twilio",
        "slack",
        "telegram",
        "mattermost",
        "facebook",
        "webexteams",
        "rocketchat",
    }


def test_create_single_input_channels():
    channels = run.create_http_input_channels("facebook", CREDENTIALS_FILE)
    assert len(channels) == 1
    assert channels[0].name() == "facebook"


def test_create_single_input_channels_by_class():
    channels = run.create_http_input_channels(
        "rasa.core.channels.channel.RestInput", CREDENTIALS_FILE
    )
    assert len(channels) == 1
    assert channels[0].name() == "rest"


def test_create_single_input_channels_by_class_wo_credentials():
    channels = run.create_http_input_channels(
        "rasa.core.channels.channel.RestInput", credentials_file=None
    )

    assert len(channels) == 1
    assert channels[0].name() == "rest"
