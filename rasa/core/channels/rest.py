import asyncio
import inspect
import json
import logging
from asyncio import CancelledError, Queue
from functools import partial
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    NoReturn,
    Optional,
    Text,
    Union,
)

import structlog
from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import BaseHTTPResponse, HTTPResponse, ResponseStream

import rasa.utils.endpoints
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    InputChannel,
    OnNewMessageType,
    UserMessage,
)

logger = logging.getLogger(__name__)
structlogger = structlog.get_logger()


class RestInput(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa and
    retrieve responses from the assistant.
    """

    @classmethod
    def name(cls) -> Text:
        return "rest"

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

        await queue.put("DONE")

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        return req.json.get("sender", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Optional[Text]:
        return req.json.get("message", None)

    def _extract_input_channel(self, req: Request) -> Text:
        return req.json.get("input_channel") or self.name()

    def get_metadata(self, request: Request) -> Optional[Dict[Text, Any]]:
        """Extracts additional information from the incoming request.

         Implementing this function is not required. However, it can be used to extract
         metadata from the request. The return value is passed on to the
         ``UserMessage`` object and stored in the conversation tracker.

        Args:
            request: incoming request with the message of the user

        Returns:
            Metadata which was extracted from the request.
        """
        return request.json.get("metadata", None)

    async def stream_response(
        self,
        on_new_message: Callable[[UserMessage], Awaitable[None]],
        text: Text,
        sender_id: Text,
        input_channel: Text,
        metadata: Optional[Dict[Text, Any]],
        resp: ResponseStream,
    ) -> None:
        """Streams response to the client.

         If the stream option is enabled, this method will be called to
         stream the response to the client

        Args:
            on_new_message: sanic event
            text: message text
            sender_id: message sender_id
            input_channel: input channel name
            metadata: optional metadata sent with the message
            resp: response stream

        Returns:
            Sanic stream
        """
        q: Queue = Queue()
        task = asyncio.ensure_future(
            self.on_message_wrapper(
                on_new_message, text, q, sender_id, input_channel, metadata
            )
        )
        while True:
            result = await q.get()
            if result == "DONE":
                break
            else:
                await resp.write(json.dumps(result) + "\n")

        await task

    async def receive_messages(
        self, request: Request, on_new_message: Callable[[UserMessage], Awaitable[None]]
    ) -> Union[BaseHTTPResponse, ResponseStream]:
        sender_id = await self._extract_sender(request)
        text = self._extract_message(request)
        should_use_stream = rasa.utils.endpoints.bool_arg(
            request, "stream", default=False
        )
        input_channel = self._extract_input_channel(request)
        metadata = self.get_metadata(request)

        if should_use_stream:
            return ResponseStream(
                partial(
                    self.stream_response,
                    on_new_message,
                    text,
                    sender_id,
                    input_channel,
                    metadata,
                ),
                content_type="text/event-stream",
            )
        else:
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
                        headers=request.headers,
                    )
                )
            except CancelledError:
                structlogger.error(
                    "rest.message.received.timeout",
                    event_info="Message processing was cancelled.",
                )
            except Exception as e:
                structlogger.exception(
                    "rest.message.received.failure",
                    event_info=f"Message processing failed. Error: {e}",
                )

            return response.json(collector.messages)

    def blueprint(self, on_new_message: OnNewMessageType) -> Blueprint:
        """Groups the collection of endpoints used by rest channel."""
        module_type = inspect.getmodule(self)
        if module_type is not None:
            module_name = module_type.__name__
        else:
            module_name = None

        custom_webhook = Blueprint(
            "custom_webhook_{}".format(type(self).__name__),
            module_name,
        )

        # noinspection PyUnusedLocal
        @custom_webhook.route("/", methods=["GET"])
        async def health(request: Request) -> HTTPResponse:
            return response.json({"status": "ok"})

        @custom_webhook.route("/webhook", methods=["POST"])
        async def receive(request: Request) -> Union[ResponseStream, BaseHTTPResponse]:
            return await self.receive_messages(request, on_new_message)

        @custom_webhook.route(
            "/cancel_background_tasks/<sender_id:str>", methods=["POST"]
        )
        async def cancel_background_tasks(
            request: Request, sender_id: str
        ) -> HTTPResponse:
            """Cancel any in-flight background processing for the given sender."""
            structlogger.debug(
                "rest.cancel_background_tasks",
                sender_id=sender_id,
            )
            try:
                agent = request.app.ctx.agent
                cancelled = agent.cancel_background_tasks(sender_id)
            except Exception as e:
                structlogger.warning(
                    "rest.cancel_background_tasks.error",
                    sender_id=sender_id,
                    error=str(e),
                )
                return response.json({"cancelled": False}, status=500)

            return response.json({"cancelled": cancelled})

        return custom_webhook


class QueueOutputChannel(CollectingOutputChannel):
    """Output channel that collects send messages in a list.

    (doesn't send them anywhere, just collects them).
    """

    # FIXME: this is breaking Liskov substitution principle
    # and would require some user-facing refactoring to address
    messages: Queue  # type: ignore[assignment]

    @classmethod
    def name(cls) -> Text:
        """Name of QueueOutputChannel."""
        return "queue"

    # noinspection PyMissingConstructor
    def __init__(self, message_queue: Optional[Queue] = None) -> None:
        super().__init__()
        self.messages = Queue() if not message_queue else message_queue

    def latest_output(self) -> NoReturn:
        raise NotImplementedError("A queue doesn't allow to peek at messages.")

    async def _persist_message(self, message: Dict[Text, Any]) -> None:
        await self.messages.put(message)

    @property
    def supports_streaming(self) -> bool:
        """QueueOutputChannel supports streaming: chunks are forwarded to the
        SSE queue immediately so the REST client receives them in real time."""
        return True

    async def send_response_chunk(
        self,
        recipient_id: Text,
        chunk: Text,
        **kwargs: Any,
    ) -> None:
        """Forward a streamed text chunk directly onto the SSE queue.

        Each call results in one NDJSON line written to the client by
        ``stream_response``, so the end user sees tokens as they arrive
        rather than waiting for the full response.

        The base-class implementation is called first so that
        ``_accumulated_streaming_text`` stays up to date.  This allows
        ``send_text_message`` to detect and suppress follow-up
        ``_utter_responses`` calls that would otherwise push the full
        assembled text onto the queue a second time.
        """
        await super().send_response_chunk(recipient_id, chunk, **kwargs)
        await self.messages.put({"recipient_id": recipient_id, "text": chunk})

    async def send_text_message(
        self,
        recipient_id: Text,
        text: Text,
        **kwargs: Any,
    ) -> None:
        """Enqueue a text message, skipping it if it duplicates streamed content.

        When an action streams its response token-by-token via
        ``send_response_chunk`` and the same assembled text is later delivered
        again through ``_utter_responses`` → ``send_response`` →
        ``send_text_message``, this guard detects the exact-text match against
        ``_accumulated_streaming_text`` and silently drops the duplicate,
        preventing the end user from receiving the same message twice in the
        SSE stream.

        Any text that does *not* match the last streamed content (e.g. a
        separate follow-up message dispatched via ``utter_message`` for
        different content) is forwarded normally.
        """
        if self._is_duplicate_of_last_streamed_response(text):
            return
        await super().send_text_message(recipient_id, text, **kwargs)
