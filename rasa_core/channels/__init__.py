from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.channels.channel import (
    InputChannel, OutputChannel,
    UserMessage,
    CollectingOutputChannel,
    RestInput)


BUILTIN_CHANNELS = {
    'facebook',
    'slack',
    'telegram',
    'mattermost',
    'twilio',
    'cmdline',
    'rasa',
    'botframework',
    'rocketchat',
}
