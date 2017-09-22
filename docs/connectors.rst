.. _connectors:

Connecting to messaging & voice platforms
=========================================

Here's how to connect your conversational AI to the outside world.

Input Channels
--------------

Input channels are defined in the ``rasa_core.channels`` module.
Currently, there is only a ``FacebookInput`` class.

A ``FacebookInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.


Facebook Messenger
------------------

Code to create a Messenger-compatible webserver looks like this:


.. testcode::

    from rasa_core.channels import HttpInputChannel
    from rasa_core.channels.facebook import FacebookInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load("examples/babi/models/policy/current", interpreter=RegexInterpreter())

    input_channel = FacebookInput(
       fb_verify="YOUR_FB_VERIFY",
       fb_secret="YOUR_FB_SECRET",
       fb_tokens=["YOUR_FB_PAGE_TOKEN"],
       debug_mode=True
    )

    # or `agent.handle_channel(...)` for synchronous handling
    agent.handle_asynchronous(HttpInputChannel(5004, "/app", input_channel))

The arguments for the ``FacebookInput`` constructor are as follows, reading the
`messenger docs <https://developers.facebook.com/docs/graph-api/webhooks>`_ probably helps too.

- ``fb_verify``  this is a token you define, and tell facebook about, to confirm your URL wants to receive webhooks
- ``fb_secret``  your app secret. this is used to check the webhook really came from facebook
- ``page_tokens``  this is a dict of ``{page_id: page_token}`` pairs, containing all the pages for which you want to handle messages.
- ``debug_mode``  ``True`` or ``False``

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving facebook messenger messages is ``/webhook``, so the example
above would listen for messages on ``/app/webhook``. This is the url you should add in the
facebook developer portal.

