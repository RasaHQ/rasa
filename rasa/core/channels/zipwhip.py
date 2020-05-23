import asyncio
import inspect
import json
from time import sleep

import requests
from asyncio import Queue, CancelledError
from typing import Any, Awaitable, Callable, Dict, Optional, Text, List

from rasa.core.channels import InputChannel, UserMessage
from rasa.core.channels.channel import QueueOutputChannel, CollectingOutputChannel, logger
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse


class ZipwhipConnector(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa Core and
    retrieve responses from the agent."""

    def __init__(self, session_key: Text) -> None:
        """
        Connector for Zipwhip SMS service

        :param session_key Session key from Zipwhip retrieved through /login endpoint
        """
        self.session_key = session_key
        super().__init__()

    @classmethod
    def name(cls) -> Text:
        return "zipwhip"

    @classmethod
    def from_credentials(cls, credentials: Optional[Dict[Text, Any]]) -> InputChannel:
        if not credentials:
            cls.raise_missing_credentials_exception()

        # pytype: disable=attribute-error
        return cls(
            credentials.get("session_token")
        )
        # pytype: enable=attribute-error

    @staticmethod
    async def on_message_wrapper(
        on_new_message: Callable[[UserMessage], Awaitable[Any]],
        text: Text,
        queue: Queue,
        sender_id: Text,
        input_channel: Text,
        metadata: Optional[Dict[Text, Any]],
    ) -> None:
        collector = QueueOutputChannel(queue)

        message = UserMessage(
            text, collector, sender_id, input_channel=input_channel, metadata=metadata
        )
        await on_new_message(message)

        await queue.put("DONE")  # pytype: disable=bad-return-type

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        return req.json.get("finalSource", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Optional[Text]:
        return req.json.get("body", None)

    def _extract_input_channel(self, req: Request) -> Text:
        return req.json.get("input_channel") or self.name()

    def stream_response(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
        text: Text,
        sender_id: Text,
        input_channel: Text,
        metadata: Optional[Dict[Text, Any]],
    ) -> Callable[[Any], Awaitable[None]]:
        async def stream(resp: Any) -> None:
            q = Queue()
            task = asyncio.ensure_future(
                self.on_message_wrapper(
                    on_new_message, text, q, sender_id, input_channel, metadata
                )
            )
            result = None  # declare variable up front to avoid pytype error
            while True:
                result = await q.get()
                if result == "DONE":
                    break
                else:
                    await resp.write(json.dumps(result) + "\n")
            await task

        return stream  # pytype: disable=bad-return-type

    def send(self, recipient: Text, message: Text, delivery_time: float = None):
        """
        Sends a (SMS) text message to one or more contacts.
        :param recipient: Phone number of recipient or comma-separated list for multiple recipients
        :param message: The message to be sent, max 600 bytes
        :param delivery_time: Pacific epoch time in milliseconds, default sends immediately
        :return: Status code of the response
        """
        payload = {
            "session": self.session_key,
            "contacts": recipient,
            "body": message,
            "scheduledDate": delivery_time,
        }

        return requests.post(
            "https://api.zipwhip.com/message/send", data=payload
        )

    def send_all(self, recipient: Text, messages: List[Dict[Text, Any]], delivery_time: int = None, delay: int = 3):
        for index, message in enumerate(messages):
            if delivery_time is not None:
                print(self.send(recipient, message.get("text"), float(delivery_time + delay * index)).text)
            else:
                sleep(delay)
                print(self.send(recipient, message.get("text")).text)

    def blueprint(
        self, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Blueprint:
        custom_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            inspect.getmodule(self).__name__,
        )

        # noinspection PyUnusedLocal
        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> HTTPResponse:
            """
            Transform Zipwhip request into Rasa expected format

            :param request: Zipwhip message receive request
            {
                "body": "Hello from mobile device",
                "bodySize": 24,
                "address": "ptn:/+12063996756",
                "finalSource": "+12063996756",
                "finalDestination": "+12068163958",
                "fingerprint": "1514465037",
                "id": 1195453017123205120,
                "contactId": 14543967707,
                "deviceId": 377265507,
                "messageType": "MO",
                "cc": null,
                "bcc": null,
                "visible": true,
                "read": false,
                "scheduledDate": null,
                "dateDeleted": null,
                "messageTransport": 5,
                "dateDelivered": null,
                "hasAttachment": false,
                "dateCreated": "2019-11-15T21:26:24+00:00",
                "deleted": false,
                "dateRead": null,
                "statusCode": 4
            }
            :return: HTTP response to return to Zipwhip
            """
            sender_id = await self._extract_sender(request)
            text = self._extract_message(request)
            print(text)

            input_channel = self._extract_input_channel(request)
            metadata = self.get_metadata(request)

            collector = CollectingOutputChannel()
            # noinspection PyBroadException
            try:
                await on_new_message(
                    UserMessage(
                        text,
                        collector,
                        sender_id,
                        input_channel=input_channel,
                        metadata=metadata,
                    )
                )
            except CancelledError:
                logger.error(
                    "Message handling timed out for "
                    "user message '{}'.".format(text)
                )
            except Exception:
                logger.exception(
                    "An exception occured while handling "
                    "user message '{}'.".format(text)
                )

            self.send_all(sender_id, collector.messages, delay=2)
            return response.json("Success.")

        return custom_webhook
