from typing import Text, Dict, List

from rasa_core.channels.channel import (
    InputChannel, OutputChannel,
    UserMessage,
    CollectingOutputChannel,
    RestInput)

# this prevents IDE's from optimizing the imports - we need to import the
# above first, otherwise we will run into import cycles
from rasa_core.channels.socketio import SocketIOInput

pass

from rasa_core.channels.botframework import BotFrameworkInput  # nopep8
from rasa_core.channels.callback import CallbackInput  # nopep8
from rasa_core.channels.console import CmdlineInput  # nopep8
from rasa_core.channels.facebook import FacebookInput  # nopep8
from rasa_core.channels.mattermost import MattermostInput  # nopep8
from rasa_core.channels.rasa_chat import RasaChatInput  # nopep8
from rasa_core.channels.rocketchat import RocketChatInput  # nopep8
from rasa_core.channels.slack import SlackInput  # nopep8
from rasa_core.channels.telegram import TelegramInput  # nopep8
from rasa_core.channels.twilio import TwilioInput  # nopep8
from rasa_core.channels.webexteams import WebexTeamsInput  # nopep8

input_channel_classes = [
    CmdlineInput, FacebookInput, SlackInput, TelegramInput, MattermostInput,
    TwilioInput, RasaChatInput, BotFrameworkInput, RocketChatInput,
    CallbackInput, RestInput, SocketIOInput, WebexTeamsInput
]  # type: List[InputChannel]

# Mapping from a input channel name to its class to allow name based lookup.
BUILTIN_CHANNELS = {
    c.name(): c
    for c in input_channel_classes}  # type: Dict[Text, InputChannel]
