# ruff: noqa: I001
from typing import Text, Dict, List, Type, Optional

from rasa.core.channels.channel import (  # noqa: F401
    InputChannel,
    OutputChannel,
    UserMessage,
    CollectingOutputChannel,
    requires_basic_auth,
)

# this prevents IDE's from optimizing the imports - we need to import the
# above first, otherwise we will run into import cycles
from rasa.core.channels.socketio import SocketIOInput
from rasa.core.channels.botframework import BotFrameworkInput
from rasa.core.channels.callback import CallbackInput
from rasa.core.channels.console import CmdlineInput
from rasa.core.channels.mattermost import MattermostInput
from rasa.core.channels.rasa_chat import RasaChatInput
from rasa.core.channels.rest import RestInput
from rasa.core.channels.rocketchat import RocketChatInput
from rasa.core.channels.voice_ready.jambonz import JambonzVoiceReadyInput
from rasa.core.channels.voice_ready.audiocodes import AudiocodesInput
from rasa.core.channels.voice_stream.browser_audio import BrowserAudioInputChannel
from rasa.core.channels.hangouts import HangoutsInput
from rasa.core.channels.voice_stream.genesys import GenesysInputChannel
from rasa.core.channels.studio_chat import StudioChatInput
from rasa.core.channels.voice_stream.audiocodes import AudiocodesVoiceInputChannel
from rasa.core.channels.voice_stream.jambonz import JambonzStreamInputChannel

# Type annotations for channels with optional dependencies
FacebookInput: Optional[Type[InputChannel]]
SlackInput: Optional[Type[InputChannel]]
TelegramInput: Optional[Type[InputChannel]]
TwilioInput: Optional[Type[InputChannel]]
TwilioVoiceInput: Optional[Type[InputChannel]]
TwilioMediaStreamsInputChannel: Optional[Type[InputChannel]]
WebexTeamsInput: Optional[Type[InputChannel]]
CVGInput: Optional[Type[InputChannel]]

# Channels with optional dependencies - import with try-except
try:
    from rasa.core.channels.facebook import FacebookInput
except ImportError:
    FacebookInput = None

try:
    from rasa.core.channels.slack import SlackInput
except ImportError:
    SlackInput = None

try:
    from rasa.core.channels.telegram import TelegramInput
except ImportError:
    TelegramInput = None

try:
    from rasa.core.channels.twilio import TwilioInput
except ImportError:
    TwilioInput = None

try:
    from rasa.core.channels.voice_ready.twilio_voice import TwilioVoiceInput
except ImportError:
    TwilioVoiceInput = None

try:
    from rasa.core.channels.voice_stream.twilio_media_streams import (
        TwilioMediaStreamsInputChannel,
    )
except ImportError:
    TwilioMediaStreamsInputChannel = None

try:
    from rasa.core.channels.webexteams import WebexTeamsInput
except ImportError:
    WebexTeamsInput = None

try:
    from rasa.core.channels.vier_cvg import CVGInput
except ImportError:
    CVGInput = None

input_channel_classes: List[Type[InputChannel]] = [
    c
    for c in [
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
        AudiocodesInput,
        CVGInput,
        JambonzVoiceReadyInput,
        TwilioMediaStreamsInputChannel,
        BrowserAudioInputChannel,
        GenesysInputChannel,
        StudioChatInput,
        AudiocodesVoiceInputChannel,
        JambonzStreamInputChannel,
    ]
    if c is not None
]

# Mapping from an input channel name to its class to allow name based lookup.
BUILTIN_CHANNELS: Dict[Text, Type[InputChannel]] = {
    c.name(): c for c in input_channel_classes
}
