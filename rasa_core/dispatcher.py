import logging
from collections import namedtuple

import typing
from typing import Text, List, Dict, Any

from rasa_core.channels import OutputChannel
from rasa_core.nlg import NaturalLanguageGenerator

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.trackers import DialogueStateTracker


class Element(dict):
    __acceptable_keys = ['title', 'item_url', 'image_url',
                         'subtitle', 'buttons']

    def __init__(self, *args, **kwargs):
        kwargs = {key: value
                  for key, value in kwargs.items()
                  if key in self.__acceptable_keys}

        super(Element, self).__init__(*args, **kwargs)


# Makes a named tuple with entries text and data
BotMessage = namedtuple("BotMessage", "text data")


class Button(dict):
    pass


class Dispatcher(object):
    """Send messages back to user"""

    def __init__(self,
                 sender_id: Text,
                 output_channel: OutputChannel,
                 nlg: NaturalLanguageGenerator) -> None:

        self.sender_id = sender_id
        self.output_channel = output_channel
        self.nlg = nlg
        self.latest_bot_messages = []

    def utter_response(self, message: Dict[Text, Any]) -> None:
        """Send a message to the client."""

        bot_message = BotMessage(text=message.get("text"),
                                 data={"elements": message.get("elements"),
                                       "buttons": message.get("buttons"),
                                       "attachment": message.get("image")})

        self.latest_bot_messages.append(bot_message)
        self.output_channel.send_response(self.sender_id, message)

    def utter_message(self, text: Text) -> None:
        """"Send a text to the output channel"""
        # Adding the text to the latest bot messages (with no data)
        bot_message = BotMessage(text=text,
                                 data=None)

        self.latest_bot_messages.append(bot_message)
        self.output_channel.send_text_message(self.sender_id, text)

    def utter_custom_message(self, *elements: Dict[Text, Any]) -> None:
        """Sends a message with custom elements to the output channel."""

        bot_message = BotMessage(text=None,
                                 data={"elements": elements})

        self.latest_bot_messages.append(bot_message)
        self.output_channel.send_custom_message(self.sender_id, elements)

    def utter_button_message(self,
                             text: Text,
                             buttons: List[Dict[Text, Any]],
                             **kwargs: Any) -> None:
        """Sends a message with buttons to the output channel."""
        # Adding the text and data (buttons) to the latest bot messages
        bot_message = BotMessage(text=text,
                                 data={"buttons": buttons})

        self.latest_bot_messages.append(bot_message)
        self.output_channel.send_text_with_buttons(self.sender_id, text,
                                                   buttons,
                                                   **kwargs)

    def utter_attachment(self, attachment: Text) -> None:
        """Send a message to the client with attachments."""
        bot_message = BotMessage(text=None,
                                 data={"attachment": attachment})

        self.latest_bot_messages.append(bot_message)
        self.output_channel.send_image_url(self.sender_id, attachment)

    # TODO: deprecate this function
    def utter_button_template(self,
                              template: Text,
                              buttons: List[Dict[Text, Any]],
                              tracker: 'DialogueStateTracker',
                              silent_fail: bool = False,
                              **kwargs: Any
                              ) -> None:
        """Sends a message template with buttons to the output channel."""

        message = self._generate_response(template,
                                          tracker,
                                          silent_fail,
                                          **kwargs)
        if not message:
            return

        if "buttons" not in message:
            message["buttons"] = buttons
        else:
            message["buttons"].extend(buttons)
        self.utter_response(message)

    def utter_template(self,
                       template: Text,
                       tracker: 'DialogueStateTracker',
                       silent_fail: bool = False,
                       **kwargs: Any
                       ) -> None:
        """"Send a message to the client based on a template."""

        message = self._generate_response(template,
                                          tracker,
                                          silent_fail,
                                          **kwargs)

        if not message:
            return

        self.utter_response(message)

    def _generate_response(self,
                           template: Text,
                           tracker: 'DialogueStateTracker',
                           silent_fail: bool = False,
                           **kwargs: Any
                           ) -> Dict[Text, Any]:
        """"Generate a response."""

        message = self.nlg.generate(template, tracker,
                                    self.output_channel.name(),
                                    **kwargs)

        if message is None and not silent_fail:
            logger.error("Couldn't create message for template '{}'."
                         "".format(template))

        return message
