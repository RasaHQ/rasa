:desc: Deploy and Run a Rasa Chat Bot on a custom chat interface

.. _custom-connectors:

Custom Connectors
=================

.. edit-link::

You can also implement your own custom channel. You can
use the ``rasa.core.channels.channel.RestInput`` class as a template.
The methods you need to implement are ``blueprint`` and ``name``. The method
needs to create a sanic blueprint that can be attached to a sanic server.

This allows you to add REST endpoints to the server that the external
messaging service can call to deliver messages.

Your blueprint should have at least the two routes: ``health`` on ``/``,
and ``receive`` on the HTTP route ``/webhook``.

The ``name`` method defines the url prefix. E.g. if your component is
named ``myio``, the webhook you can use to attach the external service is:
``http://localhost:5005/webhooks/myio/webhook`` (replacing the hostname
and port with your values).

To send a message, you would run a command like:

.. code-block:: bash

    curl -XPOST http://localhost:5000/webhooks/myio/webhook \
      -d '{"sender": "user1", "message": "hello"}' \
      -H "Content-type: application/json"

where ``myio`` is the name of your component.

If you need to use extra information from your front end in your custom
actions, you can add this information in the ``metadata`` dict of your user
message. This information will accompany the user message through the rasa
server into the action server when applicable, where you can find it stored in
the ``tracker``. Message metadata will not directly affect NLU classification
or action prediction.

Here are all the attributes of ``UserMessage``:

.. autoclass:: rasa.core.channels.UserMessage

   .. automethod:: __init__


In your implementation of the ``receive`` endpoint, you need to make
sure to call ``on_new_message(UserMessage(text, output, sender_id))``.
This will tell Rasa Core to handle this user message. The ``output``
is an output channel implementing the ``OutputChannel`` class. You can
either implement the methods for your particular chat channel (e.g. there
are methods to send text and images) or you can use the
``CollectingOutputChannel`` to collect the bot responses Core
creates while the bot is processing your messages and return
them as part of your endpoint response. This is the way the ``RestInput``
channel is implemented. For examples on how to create and use your own output
channel, take a look at the implementations of the other
output channels, e.g. the ``SlackBot`` in ``rasa.core.channels.slack``.

To use a custom channel, you need to supply a credentials configuration file
``credentials.yml`` with the command line argument ``--credentials``.
This credentials file has to contain the module path of your custom channel and
any required configuration parameters. For example, this could look like:

.. code-block:: yaml

    mypackage.MyIO:
      username: "user_name"
      another_parameter: "some value"

Here is an example implementation for an input channel that receives the messages,
hands them over to Rasa Core, collects the bot utterances, and returns
these bot utterances as the json response to the webhook call that
posted the message to the channel:

.. literalinclude:: ../../../rasa/core/channels/channel.py
   :pyobject: RestInput
