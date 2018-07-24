from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Text, List, Dict, Any, Optional, Callable, Iterable


class UserMessage(object):
    """Represents an incoming message.

     Includes the channel the responses should be sent to."""

    DEFAULT_SENDER_ID = "default"

    def __init__(self, text, output_channel=None, sender_id=None):
        # type: (Optional[Text], Optional[OutputChannel], Text) -> None
        from rasa_core.channels.direct import CollectingOutputChannel

        self.text = text

        if output_channel is not None:
            self.output_channel = output_channel
        else:
            self.output_channel = CollectingOutputChannel()

        if sender_id is not None:
            self.sender_id = sender_id
        else:
            self.sender_id = self.DEFAULT_SENDER_ID


class InputChannel(object):
    """Input channel base class.

    Collects messages from some source and puts them into the message queue."""

    @classmethod
    def name(cls):
        """Every input channel needs a name to identify it."""
        return cls.__name__

    def start_async_listening(self, message_queue):
        # type: (Dequeue) -> None
        """Start to push the incoming messages from channel into the queue."""
        raise Exception("Input channel doesn't support async listening.")

    def start_sync_listening(self, message_handler):
        # type: (Callable[[UserMessage], None]) -> None
        """Should call the message handler for every incoming message."""
        raise Exception("Input channel doesn't support sync listening.")


class OutputChannel(object):
    """Output channel base class.

    Provides sane implementation of the send methods
    for text only output channels."""

    @classmethod
    def name(cls):
        """Every output channel needs a name to identify it."""
        return cls.__name__

    def send_text_message(self, recipient_id, message):
        # type: (Text, Text) -> None
        """Send a message through this channel."""

        raise NotImplementedError("Output channel needs to implement a send "
                                  "message for simple texts.")

    def send_image_url(self, recipient_id, image_url):
        # type: (Text, Text) -> None
        """Sends an image. Default will just post the url as a string."""

        self.send_text_message(recipient_id, "Image: {}".format(image_url))

    def send_text_with_buttons(self, recipient_id, message, buttons, **kwargs):
        # type: (Text, Text, List[Dict[Text, Any]], **Any) -> None
        """Sends buttons to the output.

        Default implementation will just post the buttons as a string."""

        self.send_text_message(recipient_id, message)
        for idx, button in enumerate(buttons):
            button_msg = "{idx}: {title} ({val})".format(
                    idx=idx + 1, title=button['title'], val=button['payload'])
            self.send_text_message(recipient_id, button_msg)

    def send_custom_message(self, recipient_id, elements):
        # type: (Text, Iterable[Dict[Text, Any]]) -> None
        """Sends elements to the output.

        Default implementation will just post the elements as a string."""

        for element in elements:
            element_msg = "{title} : {subtitle}".format(
                    title=element['title'], subtitle=element['subtitle'])
            self.send_text_with_buttons(
                    recipient_id, element_msg, element['buttons'])
