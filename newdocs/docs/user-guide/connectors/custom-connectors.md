# Custom Connectors

You can also implement your own custom channel. You can
use the `rasa.core.channels.channel.RestInput` class as a template.
The methods you need to implement are `blueprint` and `name`. The method
needs to create a sanic blueprint that can be attached to a sanic server.

This allows you to add REST endpoints to the server that the external
messaging service can call to deliver messages.

Your blueprint should have at least the two routes: `health` on `/`,
and `receive` on the HTTP route `/webhook`.

The `name` method defines the url prefix. E.g. if your component is
named `myio`, the webhook you can use to attach the external service is:
`http://localhost:5005/webhooks/myio/webhook` (replacing the hostname
and port with your values).

To send a message, you would run a command like:

```
curl -XPOST http://localhost:5005/webhooks/myio/webhook \
  -d '{"sender": "user1", "message": "hello"}' \
  -H "Content-type: application/json"
```

where `myio` is the name of your component.

If you need to use extra information from your front end in your custom
actions, you can add this information in the `metadata` dict of your user
message. This information will accompany the user message through the rasa
server into the action server when applicable, where you can find it stored in
the `tracker`. Message metadata will not directly affect NLU classification
or action prediction. If you want to change the way metadata is extracted for an
existing channel, you can overwrite the function `get_metadata`. The return value
of this method will be passed to the `UserMessage`.

Here are all the attributes of `UserMessage`:


### class rasa.core.channels.UserMessage(text=None, output_channel=None, sender_id=None, parse_data=None, input_channel=None, message_id=None)
Represents an incoming message.

Includes the channel the responses should be sent to.


#### \__init__(text=None, output_channel=None, sender_id=None, parse_data=None, input_channel=None, message_id=None)
Initialize self.  See help(type(self)) for accurate signature.


* **Return type**

    `None`


In your implementation of the `receive` endpoint, you need to make
sure to call `on_new_message(UserMessage(text, output, sender_id))`.
This will tell Rasa Core to handle this user message. The `output`
is an output channel implementing the `OutputChannel` class. You can
either implement the methods for your particular chat channel (e.g. there
are methods to send text and images) or you can use the
`CollectingOutputChannel` to collect the bot responses Core
creates while the bot is processing your messages and return
them as part of your endpoint response. This is the way the `RestInput`
channel is implemented. For examples on how to create and use your own output
channel, take a look at the implementations of the other
output channels, e.g. the `SlackBot` in `rasa.core.channels.slack`.

To use a custom channel, you need to supply a credentials configuration file
`credentials.yml` with the command line argument `--credentials`.
This credentials file has to contain the module path of your custom channel and
any required configuration parameters. For example, this could look like:

```
mypackage.MyIO:
  username: "user_name"
  another_parameter: "some value"
```

Here is an example implementation for an input channel that receives the messages,
hands them over to Rasa Core, collects the bot utterances, and returns
these bot utterances as the json response to the webhook call that
posted the message to the channel:

```
class RestInput(InputChannel):
    """A custom http input channel.

    This implementation is the basis for a custom implementation of a chat
    frontend. You can customize this to send messages to Rasa Core and
    retrieve responses from the agent."""

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

        await queue.put("DONE")  # pytype: disable=bad-return-type

    async def _extract_sender(self, req: Request) -> Optional[Text]:
        return req.json.get("sender", None)

    # noinspection PyMethodMayBeStatic
    def _extract_message(self, req: Request) -> Optional[Text]:
        return req.json.get("message", None)

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
            sender_id = await self._extract_sender(request)
            text = self._extract_message(request)
            should_use_stream = rasa.utils.endpoints.bool_arg(
                request, "stream", default=False
            )
            input_channel = self._extract_input_channel(request)
            metadata = self.get_metadata(request)

            if should_use_stream:
                return response.stream(
                    self.stream_response(
                        on_new_message, text, sender_id, input_channel, metadata
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
                return response.json(collector.messages)

        return custom_webhook
```
