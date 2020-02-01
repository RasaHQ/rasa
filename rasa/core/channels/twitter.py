from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
from rasa.core import utils
import hashlib, hmac, base64, os, json
from sanic import Blueprint, response
from sanic.request import Request
from typing import Text, List, Dict, Any, Callable, Awaitable, Iterable, Optional
from rasa.core.channels.channel import UserMessage, OutputChannel, InputChannel
from sanic.response import HTTPResponse
from TwitterAPI import TwitterAPI
logger = logging.getLogger(__name__)
class TwitterOutputChannel(OutputChannel):
    def __init__(self,messenger_client) -> None:
        self.messages = []
        self.messenger_client = messenger_client
        super().__init__()
    @classmethod
    def name(cls) -> Text:
        return "twitter"
    def send(self, message) -> None:
        self.messenger_client.request('direct_messages/events/new', message)
    async def send_text_message(
        self, recipient_id: Text, text: Text, **kwargs: Any
    ) -> None:
        for message_part in text.split("\n\n"):
            reply = json.dumps({
                                "event":{
                                        "type":"message_create",
                                        "message_create":{
                                                        "target":{
                                                            "recipient_id": recipient_id
                                                                    },"message_data":{
                                                                    "text":message_part
                                                        }
                                                    }
                                            }
                                })
            self.send(message=reply)
    async def send_image_url(
        self, recipient_id: Text, image: Text, **kwargs: Any
    ) -> None:
        """Sends an image. Default will just post the url as a string."""
        reply = json.dumps({
                            "event":{
                                        "type":"message_create","message_create":{
                                                                "target":{
                                                                    "recipient_id": recipient_id
                                                                            },"message_data":{
                                                                            "text":image
                                                                }
                                                            }
                                                }
                                    })
        self.send(message=reply)
    async def send_attachment(
        self, recipient_id: Text, attachment: Text, **kwargs: Any
    ) -> None:
        """Sends an attachment. Default will just post as a string."""
        reply = json.dumps({
                            "event":{
                                        "type":"message_create","message_create":{
                                                                "target":{
                                                                    "recipient_id": recipient_id
                                                                            },"message_data":{
                                                                            "text":attachment
                                                                }
                                                            }
                                                }
                                    })
        self.send(message=reply)
    async def send_text_with_buttons(
        self,
        recipient_id: Text,
        text: Text,
        buttons: List[Dict[Text, Any]],
        **kwargs: Any,
    ) -> None:
        reply = {
                    "event": {
                        "type": "message_create",
                        "message_create": {
                        "target": {
                            "recipient_id": recipient_id
                        },
                        "message_data": {
                            "text": text,
                            "quick_reply": {
                            "type": "options"
                            }
                        }
                        }
                    }
            }
        button_list = []
        for button in buttons:
            option = {
                                "label": button['title'],
                                "description": button['payload']
                    }
            button_list.append(option)
        reply['event']['message_create']['message_data']['quick_reply']['options'] = button_list
        self.send(message=json.dumps(reply))
    async def send_custom_json(
        self, recipient_id: Text, json_message: Dict[Text, Any], **kwargs: Any
    ) -> None:
        reply = json.dumps({
                            "event":{
                                        "type":"message_create","message_create":{
                                                                "target":{
                                                                    "recipient_id": recipient_id
                                                                            },"message_data":{
                                                                            "text":json_message
                                                                }
                                                            }
                                                }
                                    })
        self.send(message=reply)
class TwitterInput(InputChannel):
    @classmethod
    def name(cls):
        return "twitter"
    @classmethod
    def from_credentials(cls, credentials):
        if not credentials:
            cls.raise_missing_credentials_exception()
        return cls(credentials.get("consumer_key"),
                   credentials.get("consumer_secret"),
                   credentials.get("access_token"),
                   credentials.get("access_token_secret"),
                   credentials.get("screen_name")
                   )
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret,screen_name):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_token_secret = access_token_secret
        self.screen_name = screen_name
    def initObject(self):
        twitterAPI = TwitterAPI(self.consumer_key, self.consumer_secret, self.access_token, self.access_token_secret)
        return twitterAPI
    def blueprint(self, on_new_message: Callable[[UserMessage], Awaitable[Any]]):
        twitter_webhook = Blueprint('twitter_webhook', __name__)
        @twitter_webhook.route("/", methods=['GET'])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})
        @twitter_webhook.route("/webhook", methods=['GET'])
        async def verification(request: Request) -> HTTPResponse:
            crc = request.args['crc_token']
            validation = hmac.new(
                key=bytes(self.consumer_secret, 'utf-8'),
                msg=bytes(crc[0], 'utf-8'),
                digestmod = hashlib.sha256
                )
            digested = base64.b64encode(validation.digest())
            return response.json({
                    'response_token': 'sha256=' + format(str(digested)[2:-1])
            })			
        @twitter_webhook.route("/webhook", methods=['POST'])
        async def webhook(request: Request) -> HTTPResponse:
            requestJson = request.json
            api = self.initObject() 
            if 'direct_message_events' in requestJson.keys():
                r = api.request('users/show',{'screen_name':self.screen_name})
                eventType = requestJson['direct_message_events'][0].get("type")
                messageObject = requestJson['direct_message_events'][0].get('message_create', {})
                messageSenderId = messageObject.get('sender_id')  
                if eventType != 'message_create':
                    return response.text(" ")
                if messageSenderId != r.json()['id_str']:
                    text = messageObject.get('message_data').get('text')
                    sender_id = messageSenderId
                    out = TwitterOutputChannel(api)			
                    await on_new_message(UserMessage(text, out, sender_id,input_channel=self.name()))
                    return response.text("success")
                else:
                    return response.text(" ")
            else:
                return response.text(" ")
        return twitter_webhook
