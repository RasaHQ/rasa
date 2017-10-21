from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from typing import Text, List, Dict, Any

from rasa_core.domain import Domain
from rasa_core.channels import OutputChannel


class Element(dict):
    __acceptable_keys = ['title', 'item_url', 'image_url',
                         'subtitle', 'buttons']

    def __init__(self, *args, **kwargs):
        kwargs = {key: value
                  for key, value in kwargs.items()
                  if key in self.__acceptable_keys}

        super(Element, self).__init__(*args, **kwargs)


class Button(dict):
    # TODO: Decide if this should do more
    pass


class Dispatcher(object):
    """Send messages back to user"""

    def __init__(self, sender, output_channel, domain):
        # type: (Text, OutputChannel, Domain) -> None

        self.sender = sender
        self.output_channel = output_channel
        self.domain = domain
        self.send_messages = []

    def utter_message(self, message):
        # type: (Dict[Text, Any]) -> None
        """Send a message to the client."""

        if message.get("buttons"):
            self.utter_button_message(message.get("text"),
                                      message.get("buttons"))
        elif message.get("image"):
            self.utter_text(message.get("text"))
            self.utter_attachment(message.get("image"))
        else:
            self.utter_text(message.get("text"))

    def utter_text(self, text):
        if self.sender is not None and self.output_channel is not None:
            for message_part in text.split("\n\n"):
                self.output_channel.send_text_message(self.sender, message_part)
                self.send_messages.append(message_part)

    def utter_custom_message(self, *elements):
        # type: (*Dict[Text, Any]) -> None
        """Sends a message with custom elements to the output channel."""

        self.output_channel.send_custom_message(self.sender, elements)

    def utter_button_message(self, text, buttons, **kwargs):
        # type: (Text, List[Dict[Text, Any]], **Any) -> None
        """Sends a message with buttons to the output channel."""

        self.output_channel.send_text_with_buttons(self.sender, text, buttons,
                                                   **kwargs)

    def utter_attachment(self, attachment):
        # type: (Text) -> None
        """Send a message to the client with attachements."""

        self.output_channel.send_image_url(self.sender, attachment)

    def utter_button_template(self, template, buttons, **kwargs):
        # type: (Text, List[Dict[Text, Any]], **Any) -> None
        """Sends a message template with buttons to the output channel."""

        t = self.retrieve_template(template, **kwargs)
        if "buttons" not in t:
            t["buttons"] = buttons
        else:
            t["buttons"].extend(buttons)
        self.utter_message(t)

    def utter_template(self, template, **kwargs):
        # type: (Text, **Any) -> None
        """"Send a message to the client based on a template."""

        self.utter_message(self.retrieve_template(template, **kwargs))

    def retrieve_template(self, template, **kwargs):
        # type: (Text, **Any) -> Dict[Text, Any]
        """Retrieve a named template from the domain."""

        r = self.domain.random_template_for(template)
        if r is not None:
            if len(kwargs) > 0:
                r["text"] = r["text"].format(**kwargs)
            return r
        else:
            return {"text": "Undefined utter template <{}>".format(template)}
