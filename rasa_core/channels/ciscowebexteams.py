from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
from typing import Text, Optional, List

from flask import Blueprint, request, jsonify, make_response, Response
from webexteamssdk import WebexTeamsAPI, Webhook

from rasa_core.channels import InputChannel
from rasa_core.channels.channel import UserMessage, OutputChannel

logger = logging.getLogger(__name__)


"""
class CiscoWebexTeamsBot(WebexTeamsAPI, OutputChannel):
    #A Cisco Webex Teams communication channel

    @classmethod
    def name(cls):
        return "ciscowebexteams"

    def __init__(self, access_token, ciscowebexteams_room):

        self.ciscowebexteams_room = ciscowebexteams_room
        super(CiscoWebexTeamsBot, self).__init__(access_token)


    def send_text_message(self, recipient_id, message):
        recipient = self.ciscowebexteams_room or recipient_id
        for message_part in message.split("\n\n"):
            super(CiscoWebexTeamsBot, self).messages.create(roomId=recipient,text=message_part)

    def send_image_url(self, recipient_id, image_url, message=""):
        recipient = self.ciscowebexteams_room or recipient_id
        return super(CiscoWebexTeamsBot, self).messages.create(
                                              roomId=recipient,
                                              files=[image_url])
    def send_file_url(self, recipient_id, file_url, message=""):
        #file url can be public url or local file path
        recipient = self.ciscowebexteams_room or recipient_id
        return super(CiscoWebexTeamsBot, self).messages.create(
                                              roomId=recipient,
                                              files=[file_url])
"""


class CiscoWebexTeamsBot(OutputChannel):
    #A Cisco Webex Teams communication channel
    #an alternate implementation that doesn't subclass WebexTeamsAPI - seems more readable
    @classmethod
    def name(cls):
        return "ciscowebexteams"

    def __init__(self, access_token, ciscowebexteams_room):

        self.ciscowebexteams_room = ciscowebexteams_room
        self.api = WebexTeamsAPI(access_token)


    def send_text_message(self, recipient_id, message):
        recipient = self.ciscowebexteams_room or recipient_id
        for message_part in message.split("\n\n"):
            self.api.messages.create(roomId=recipient, text=message_part)

    def send_image_url(self, recipient_id, image_url, message=""):
        recipient = self.ciscowebexteams_room or recipient_id
        return self.api.messages.create(roomId=recipient,files=[image_url])

    def send_file_url(self, recipient_id, file_url, message=""):
        recipient = self.ciscowebexteams_room or recipient_id
        return self.api.messages.create(roomId=recipient,files=[file_url])



class CiscoWebexTeamsInput(InputChannel):
    """Webex Teams input channel implementation. Based on the HTTPInputChannel."""

    @classmethod
    def name(cls):
        return "ciscowebexteams"

    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()

        #    dont think we need to add credentials.get("CiscoWebexTeams_Room")
        return cls(credentials.get("ciscowebexteams_accesstoken"),
                    credentials.get("ciscowebexteams_room"))

    def __init__(self, ciscowebexteams_accesstoken, ciscowebexteams_room):
        # type: (Text, Optional[Text], Optional[List[Text]]) -> None
        """Create a Cisco Webex Teams input channel.

        Needs a couple of settings to properly authenticate and validate
        messages. Details here https://developer.webex.com/authentication.html
        :param ciscowebexteams_accesstoken: Your Cisco Webex Teams Authentication token.

        :param ciscowebexteams_room: the string identifier for a room to which
            the bot posts
        """
        self.ciscowebexteams_token = ciscowebexteams_accesstoken
        self.ciscowebexteams_room = ciscowebexteams_room
        self.api = WebexTeamsAPI(ciscowebexteams_accesstoken)

    def process_message(self, on_new_message, text, sender_id):

        try:
            out_channel = CiscoWebexTeamsBot(self.ciscowebexteams_token, self.ciscowebexteams_room)
            user_msg = UserMessage(text, out_channel, sender_id,
                                   input_channel=self.name())
            on_new_message(user_msg)
        except Exception as e:
            logger.error("Exception when trying to handle "
                         "message.{0}".format(e))
            logger.error(str(e), exc_info=True)

        return make_response()

    def blueprint(self, on_new_message):
        ciscowebexteams_webhook = Blueprint('ciscowebexteams_webhook', __name__)

        @ciscowebexteams_webhook.route("/", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @ciscowebexteams_webhook.route("/webhook", methods=['GET', 'POST'])
        def webhook():
            """Processes incoming requests to the '/webhook' URI."""
            if request.method == 'GET':
                return ("""<!DOCTYPE html>
                   <html lang="en">
                       <head>
                           <meta charset="UTF-8">
                           <title>Webex Teams Bot served via Flask</title>
                       </head>
                   <body>
                   <p>
                   <strong>Your Flask web server is up and running!</strong>
                   </p>
                   <p>
                   Hey there. I am your Rasa Bot
                   </p>
                   </body>
                   </html>
                """)
            elif request.method == 'POST':
                """Respond to inbound webhook JSON HTTP POST from Webex Teams."""

                # Get the POST data sent from Webex Teams
                json_data = request.json
                print("\n")
                print("WEBHOOK POST RECEIVED:")
                print(json_data)
                print("\n")

                # Create a Webhook object from the JSON data
                webhook_obj = Webhook(json_data)
                # Get the room details
                room = self.api.rooms.get(webhook_obj.data.roomId)
                # Get the message details
                message = self.api.messages.get(webhook_obj.data.id)
                # Get the sender's details
                person = self.api.people.get(message.personId)

                print("NEW MESSAGE IN ROOM '{}'".format(room.title))
                print("FROM '{}'".format(person.displayName))
                print("MESSAGE '{}'\n".format(message.text))

                # This is a VERY IMPORTANT loop prevention control step.
                # If you respond to all messages...  You will respond to the messages
                # that the bot posts and thereby create a loop condition.
                me = self.api.people.me()
                if message.personId == me.id:
                    # Message was sent by me (bot); do not respond.
                    return 'OK'

                else:
                    return self.process_message(
                            on_new_message,
                            text=message.text,
                            sender_id=message.personId)


            return make_response()

        return ciscowebexteams_webhook
