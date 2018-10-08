from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

from rasa_core.channels.botframework import BotFrameworkInput
from rasa_core.channels.callback import CallbackInput
from rasa_core.channels.console import CmdlineInput
from rasa_core.channels.facebook import FacebookInput
from rasa_core.channels.mattermost import MattermostInput
from rasa_core.channels.rasa_chat import RasaChatInput
from rasa_core.channels.rocketchat import RocketChatInput
from rasa_core.channels.slack import SlackInput
from rasa_core.channels.telegram import TelegramInput
from rasa_core.channels.twilio import TwilioInput

input_channel_classes = [
    CmdlineInput, FacebookInput, SlackInput, TelegramInput, MattermostInput,
    TwilioInput, RasaChatInput, BotFrameworkInput, RocketChatInput,
    CallbackInput, RestInput, SocketIOInput
]  # type: List[InputChannel]

# Mapping from a input channel name to its class to allow name based lookup.
BUILTIN_CHANNELS = {
    c.name(): c
    for c in input_channel_classes}  # type: Dict[Text, InputChannel]
