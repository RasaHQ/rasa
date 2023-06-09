from typing import Text, Dict, List, Type

from rasa.core.channels.channel import (  # noqa: F401
    InputChannel,
    OutputChannel,
    UserMessage,
    CollectingOutputChannel,
)

# this prevents IDE's from optimizing the imports - we need to import the
# above first, otherwise we will run into import cycles
from rasa.core.channels.socketio import SocketIOInput
from rasa.core.channels.botframework import BotFrameworkInput
from rasa.core.channels.callback import CallbackInput
from rasa.core.channels.console import CmdlineInput
from rasa.core.channels.facebook import FacebookInput
from rasa.core.channels.mattermost import MattermostInput
from rasa.core.channels.rasa_chat import RasaChatInput
from rasa.core.channels.rest import RestInput
from rasa.core.channels.rocketchat import RocketChatInput
from rasa.core.channels.slack import SlackInput
from rasa.core.channels.telegram import TelegramInput
from rasa.core.channels.twilio import TwilioInput
from rasa.core.channels.twilio_voice import TwilioVoiceInput
from rasa.core.channels.webexteams import WebexTeamsInput
from rasa.core.channels.hangouts import HangoutsInput

input_channel_classes: List[Type[InputChannel]] = [
    CmdlineInput,
    FacebookInput,
    SlackInput,
    TelegramInput,
    MattermostInput,
    TwilioInput,
    TwilioVoiceInput,
    RasaChatInput,
    BotFrameworkInput,
    RocketChatInput,
    CallbackInput,
    RestInput,
    SocketIOInput,
    WebexTeamsInput,
    HangoutsInput,
]

# Mapping from an input channel name to its class to allow name based lookup.
BUILTIN_CHANNELS: Dict[Text, Type[InputChannel]] = {
    c.name(): c for c in input_channel_classes
}
