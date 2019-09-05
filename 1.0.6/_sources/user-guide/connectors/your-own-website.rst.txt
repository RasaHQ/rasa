:desc: Deploy and Run a Rasa Chat Bot on a Website

.. _your-own-website:

Your Own Website
================

If you just want an easy way for users to test your bot, the best option
is usually the chat interface that ships with Rasa X, where you can `invite users
to test your bot <../../rasa-x/docs/get-feedback-from-test-users>`_.

If you already have an existing website and want to add a Rasa assistant to it,
you have a couple of options:

- `Rasa Webchat <https://github.com/mrbot-ai/rasa-webchat>`_, which
  uses websockets.
- `Chatroom <https://github.com/scalableminds/chatroom>`_, which
  uses regular HTTP requests.

Websocket Channel
~~~~~~~~~~~~~~~~~

The SocketIO channel uses websockets and is real-time. You need to supply
a ``credentials.yml`` with the following content:

.. code-block:: yaml

   socketio:
     user_message_evt: user_uttered
     bot_message_evt: bot_uttered
     session_persistence: true/false

The first two configuration values define the event names used by Rasa Core
when sending or receiving messages over socket.io.

By default, the socketio channel uses the socket id as ``sender_id``, which causes
the session to restart at every page reload. ``session_persistence`` can be
set to ``true`` to avoid that. In that case, the frontend is responsible
for generating a session id and sending it to the Rasa Core server by
emitting the event ``session_request`` with ``{session_id: [session_id]}``
immediately after the ``connect`` event.

The example `Webchat <https://github.com/mrbot-ai/rasa-webchat>`_
implements this session creation mechanism (version >= 0.5.0).


.. _rest_channels:

REST Channels
~~~~~~~~~~~~~


The ``RestInput`` and ``CallbackInput`` channels can be used for custom integrations.
They provide a URL where you can post messages and either receive response messages
directly, or asynchronously via a webhook.


RestInput
^^^^^^^^^

The ``rest`` channel will provide you with a REST endpoint to post messages
to and in response to that request will send back the bots messages.
Here is an example on how to connect the ``rest`` input channel
using the run script:

.. code-block:: bash

 rasa run

you need to ensure your ``credentials.yml`` has the following content:

.. code-block:: yaml

   rest:
     # you don't need to provide anything here - this channel doesn't
     # require any credentials

After connecting the ``rest`` input channel, you can post messages to
``POST /webhooks/rest/webhook`` with the following format:

.. code-block:: json

   {
     "sender": "Rasa",
     "message": "Hi there!"
   }

The response to this request will include the bot responses, e.g.

.. code-block:: json

   [
     {"text": "Hey Rasa!"}, {"image": "http://example.com/image.jpg"}
   ]


CallbackInput
^^^^^^^^^^^^^

The ``callback`` channel behaves very much like the ``rest`` input,
but instead of directly returning the bot messages to the HTTP
request that sends the message, it will call a URL you can specify
to send bot messages.

Here is an example on how to connect the
``callback`` input channel using the run script:

.. code-block:: bash

 rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   callback:
     # URL to which Core will send the bot responses
     url: "http://localhost:5034/bot"

After connecting the ``callback`` input channel, you can post messages to
``POST /webhooks/callback/webhook`` with the following format:

.. code-block:: json

   {
     "sender": "Rasa",
     "message": "Hi there!"
   }

The response will simply be ``success``. Once Core wants to send a
message to the user, it will call the URL you specified with a ``POST``
and the following ``JSON`` body:

.. code-block:: json

   [
     {"text": "Hey Rasa!"}, {"image": "http://example.com/image.jpg"}
   ]
