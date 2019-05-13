:desc: Choose Facebook, Slack or Telegram as your channel for contextual
       Assistants and chatbots or build new ones for your own app or website
       using open source Rasa Stack. 

.. _connectors:

Chat & Voice platforms
======================

Here's how to connect your conversational AI to the outside world.

See :ref:`here <custom_channels>` for details on how to build a
custom input channel.

.. contents::

Input channels are defined in the ``rasa.core.channels`` module.
Currently, there is code for connecting to
facebook, slack, telegram, mattermost and twilio. If the connection
you want is missing, this is a great place to start contributing!

If you're testing on your local machine (e.g. not a server), you
will need to use ngrok_. This gives your machine a domain name
and so that facebook, slack, etc. know where to send messages.

Managing Credentials
--------------------

To connect to most channels, you will need to add some credentials
(e.g. an API token). Rasa Core will read these properties from a
credentials file in yaml format, here is an example containing most
channels:

.. literalinclude:: ../../examples/moodbot/credentials.yml

.. note::

    You only need to include the channels you want to connect to in
    your ``credentials.yml``.


.. _facebook_connector:

Facebook Setup
--------------

You first need to retrieve some credentials to connect to the
Facebook Messenger. Once you have them you can
**either** attach the input channel running the ``rasa run``
command, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the FB credentials:**
You need to set up a Facebook app and a page.

  1. To create the app head over to
     `Facebook for Developers <https://developers.facebook.com/>`_
     and click on *My Apps* -> *Add New App*.
  2. Go onto the dashboard for the app and under *Products*,
     find the *Messenger* section and click *Set Up*. Scroll down to
     *Token Generation* and click on the link to create a new page for your
     app.
  3. Create your page and select it in the dropdown menu for the
     *Token Generation*. The shown *Page Access Token* is the
     ``page-access-token`` needed later on.
  4. Locate the *App Secret* in the app dashboard under *Settings* -> *Basic*.
     This will be your ``secret``.
  5. Use the collected ``secret`` and ``page-access-token`` in your
     ``credentials.yml``, and add a field called ``verify`` containing
     a string of your choice. Start ``rasa run`` with the
     ``--credentials credentials.yml`` option.
  6. Set up a *Webhook* and select at least the *messaging* and
     *messaging_postback* subscriptions. Insert your callback URL which will
     look like ``https://<YOUR_HOST>/webhooks/facebook/webhook``. Insert the
     *Verify Token* which has to match the ``verify``
     entry in your ``credentials.yml``.


For more detailed steps, visit the
`messenger docs <https://developers.facebook.com/docs/graph-api/webhooks>`_.


Using the run script
^^^^^^^^^^^^^^^^^^^^
If you want to connect to facebook using the run script, e.g. using:

.. code-block:: bash

  rasa run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

  facebook:
    verify: "rasa-bot"
    secret: "3e34709d01ea89032asdebfe5a74518"
    page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"

The endpoint for receiving facebook messenger messages is
``http://localhost:5005/webhooks/facebook/webhook``, replacing
the host and port with the appropriate values. This is the URL
you should add in the configuration of the webhook.

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``FacebookInput`` instance provides a sanic blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_facebook_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE


.. _webexteams_connector:

Cisco Webex Teams Setup
-----------------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the ``rasa run``
command, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Cisco Webex Teams credentials:**

You need to set up a bot. Please visit below link to create a bot
`Webex Authentication <https://developer.webex.com/authentication.html>`_.

After you have created the bot through Cisco Webex Teams, you need to create a
room in Cisco Webex Teams. Then add the bot in the room the same way you would
add a person in the room.

You need to note down the room ID for the room you created. This room ID will
be used in ``room`` variable in the ``credentials.yml`` file.

Please follow this link below to find the room ID
``https://developer.webex.com/endpoint-rooms-get.html``

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the ``webexteams`` input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   webexteams:
     access_token: "YOUR-BOT-ACCESS-TOKEN"
     room: "YOUR-CISCOWEBEXTEAMS-ROOM-ID"


The endpoint for receiving Cisco Webex Teams messages is
``http://localhost:5005/webhooks/webexteams/webhook``, replacing
the host and port with the appropriate values. This is the URL
you should add in the OAuth & Permissions section.

.. note::

   If you do not set the ``room`` keyword
   argument, messages will by delivered back to
   the user who sent them.

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``WebexTeamsInput`` instance provides a sanic blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a WebexTeams-compatible webserver looks like this:


.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_webexteams_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

.. _slack_connector:

Slack Setup
-----------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the ``rasa run``
command, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Slack credentials:** You need to set up a Slack app.

  1. To create the app go to: https://api.slack.com/apps and click
     on *"Create New App"*.
  2. Activate the following features: interactive components, event
     subscriptions, bot users, permissions (for basic functionality
     you should subscribe to the ``message.channel``,
     ``message.groups``, ``message.im`` and ``message.mpim`` events)
  3. The ``slack_channel`` is the target your bot posts to.
     This can be a channel, an app or an individual person
  4. Use the entry for ``Bot User OAuth Access Token`` in the
     "OAuth & Permissions" tab as your ``slack_token``


For more detailed steps, visit the
`slack api docs <https://api.slack.com/incoming-webhooks>`_.

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the slack input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   slack:
     slack_token: "xoxb-286425452756-safjasdf7sl38KLls"
     slack_channel: "#my_channel" <!-- or "@my_app" -->


The endpoint for receiving slack messages is
``http://localhost:5005/webhooks/slack/webhook``, replacing
the host and port with the appropriate values. This is the URL
you should add in the OAuth & Permissions section.

.. note::

   If you do not set the slack_channel keyword
   argument, messages will by delivered back to
   the user who sent them.

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``SlackInput`` instance provides a sanic blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_slack_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

.. _mattermost_connector:

Mattermost Setup
----------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the ``rasa run``
command, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to setup the outgoing webhook:**

   1. To create the mattermost outgoing webhook login to your mattermost
      team site and go to **Main Menu > Integrations > Outgoing Webhooks**
   2. Click **Add outgoing webhook**
   3. Fill out the details including the channel you want the bot in.
      You will need to ensure the **trigger words** section is setup
      with @yourbotname so that way it doesn't trigger on everything
      that is said.
   4. Make sure **trigger when** section is set to value
      **first word matches a trigger word exactly**
   5. For the callback url this needs to be your ngrok url where you
      have your webhook running in core or your public address, e.g.
      ``http://test.example.com/webhooks/mattermost/webhook``


For more detailed steps, visit the
`mattermost docs at <https://docs.mattermost.com/guides/developer.html>`_.

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the mattermost input channel using the
run script, e.g. using:

.. code-block:: bash

 rasa core run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   mattermost:
     url: "https://chat.example.com/api/v4"
     team: "community"
     user: "user@user.com"
     pw: "password"


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``MattermostInput`` instance provides a sanic blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Mattermost-compatible webserver looks like this:

.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_mattermost_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

The arguments for the ``handle_channels`` are the input channels and
the port. The endpoint for receiving mattermost channel messages
is ``/webhooks/mattermost/webhook``. This is the url you should
add in the mattermost outgoing webhook.

.. _telegram_connector:

Telegram Setup
--------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the ``rasa run``
command, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Telegram credentials:**
You need to set up a Telegram bot.

  1. To create the bot, go to: https://web.telegram.org/#/im?p=@BotFather,
     enter */newbot* and follow the instructions.
  2. At the end you should get your ``access_token`` and the username you
     set will be your ``verify``.
  3. If you want to use your bot in a group setting, it's advisable to
     turn on group privacy mode by entering */setprivacy*. Then the bot
     will only listen when the message is started with */bot*

For more information on the Telegram HTTP API, go to
https://core.telegram.org/bots/api

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to telegram using the run script, e.g. using:

.. code-block:: bash

  rasa run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   telegram:
     access_token: "490161424:AAGlRxinBRtKGb21_rlOEMtDFZMXBl6EC0o"
     verify: "your_bot"
     webhook_url: "your_url.com/webhooks/telegram/webhook"


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``TelegramInput`` instance provides a sanic blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:

.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_telegram_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

The arguments for the ``handle_channels`` are the input channels and
the port. The endpoint for receiving telegram channel messages
is ``/webhooks/telegram/webhook``. This is the URL you should use
as the ``webhook_url``. To get the bot to listen for messages at
that URL, go to ``myurl.com/webhooks/telegram/set_webhook``
first to set the webhook.

.. _twilio_connector:

Twilio Setup
--------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the provided ``rasa run``
script, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Twilio credentials:**
You need to set up a Twilio account.

  1. Once you have created a Twilio account, you need to create a new
     project. The basic important product to select here
     is ``Programmable SMS``.
  2. Once you have created the project, navigate to the Dashboard of
     ``Programmable SMS`` and click on ``Get Started`` and follow the
     steps to connect a phone number to the project.
  3. Now you can use the ``Account SID``, ``Auth Token`` and the phone
     number you purchased in your credentials yml.

For more information on the Twilio REST API, go to
https://www.twilio.com/docs/iam/api

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to the twilio input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa core run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   twilio:
     account_sid: "ACbc2dxxxxxxxxxxxx19d54bdcd6e41186"
     auth_token: "e231c197493a7122d475b4xxxxxxxxxx"
     twilio_number: "+440123456789"

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``TwilioInput`` instance provides a sanic blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Twilio-compatible webserver looks like this:

.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_twilio_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

The arguments for the ``handle_channels`` are the input channels and
the port. The endpoint for receiving twilio channel messages
is ``/webhooks/twilio/webhook``.


.. _rocketchat_connector:

RocketChat Setup
----------------

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to set up Rocket.Chat:**

 1. Create a user that will be used to post messages and set its
    credentials at credentials file.
 2. Create a Rocket.Chat outgoing webhook by logging as admin to
    Rocket.Chat and going to
    **Administration > Integrations > New Integration**.
 3. Select **Outgoing Webhook**.
 4. Set **Event Trigger** section to value **Message Sent**.
 5. Fill out the details including the channel you want the bot
    listen to. Optionally, it is possible to set the
    **Trigger Words** section with ``@yourbotname`` so that way it
    doesn't trigger on everything that is said.
 6. Set your **URLs** section to the Rasa URL where you have your
    webhook running, in Core or your public address with
    ``/webhooks/rocketchat/webhook``, e.g.:
    ``http://test.example.com/webhooks/rocketchat/webhook``.

For more information on the Rocket.Chat Webhooks, go to
https://rocket.chat/docs/administrator-guides/integrations/


Using run script
^^^^^^^^^^^^^^^^

If you want to connect to the rocketchat input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   rocketchat:
     user: "yourbotname"
     password: "YOUR_PASSWORD"
     server_url: "https://demo.rocket.chat"

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``RocketChatInput`` instance provides a sanic blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a RocketChat-compatible webserver looks like this:

.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_rocketchat_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

The arguments for the ``handle_channels`` are the input channels and
the port. The endpoint for receiving mattermost channel messages
is ``/webhooks/rocketchat/webhook``. This is the url you should add in the
RocketChat outgoing webhook.

.. _botframework_connector:

Microsoft Bot Framework Setup
-----------------------------

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the botframework input channel using the
run script, e.g. using:

.. code-block:: bash

 rasa run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   botframework:
     app_id: "MICROSOFT_APP_ID"
     app_password: "MICROSOFT_APP_PASSWORD"

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``BotFrameworkInput`` instance provides a sanic blueprint for creating
a webserver. This lets you seperate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Microsoft Bot Framework-compatible webserver looks like this:

.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_botframework_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

The arguments for the ``handle_channels`` are the input channels and
the port. The endpoint for receiving botframework channel messages
is ``/webhooks/botframework/webhook``. This is the url you should
add in your microsoft bot service configuration.

.. _socketio_connector:

SocketIO Setup
--------------

You can **either** attach the input channel running the
``rasa run`` command, or you can attach the channel in your
own code.

Using run script
^^^^^^^^^^^^^^^^

If you want to connect the socketio input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   socketio:
     user_message_evt: user_uttered
     bot_message_evt: bot_uttered
     session_persistence: true/false

The first two configuration values define the event names used by Rasa Core
when sending or receiving messages over socket.io.

By default, the socketio channel uses the socket id as sender_id which causes
the session to restart at every page reload. ``session_persistence`` can be
set to ``true`` to avoid that. In that case, the frontend is responsible
for generating a session id and sending it to the Rasa Core server by
emitting the event ``session_request`` with ``{session_id: [session_id]}``
immediately after the ``connect`` event.

The example `Webchat <https://github.com/mrbot-ai/rasa-webchat>`_
implements this session creation mechanism (version >= 0.5.0).

Directly using python
^^^^^^^^^^^^^^^^^^^^^

Code to create a Socket.IO-compatible webserver looks like this:

.. literalinclude:: ../../tests/core/test_channels.py
   :pyobject: test_socketio_channel
   :start-after: START DOC INCLUDE
   :dedent: 8
   :end-before: END DOC INCLUDE

The arguments for the ``handle_channels`` are the input channels and
the port. Once started, you should be able to connect to
``http://localhost:5005`` with your socket.io client.

.. _ngrok:

Using Ngrok For Local Testing
-----------------------------

You can use https://ngrok.com/ to create a local webhook from your
machine that is Publicly available on the internet so you can use it
with applications like Slack, Facebook, etc.

The command to run a ngrok instance for port 5002 for example would be:

.. code-block:: bash

  ngrok httpd 5002

**Ngrok is only needed if you don't have a public IP and are testing locally**

ngrok will create a https address for your computer, for example
https://xxxxxx.ngrok.io . For a facebook bot, your webhook address
would then be https://xxxxxx.ngrok.io/webhooks/facebook/webhook,
for telegram https://xxxxxx.ngrok.io/webhooks/telegram/webhook, etc.

.. _rest_channels:

REST Channels
-------------

If you want to put a widget on your website so that people can
talk to your bot, check out these two projects:

- `Rasa Webchat <https://github.com/mrbot-ai/rasa-webchat>`_
  uses sockets.
- `Chatroom <https://github.com/scalableminds/chatroom>`_
  uses regular HTTP requests.

For these use cases it is easiest to use either the ``RestInput`` or the
``CallbackInput`` channels. They will provide you with a URL to post the
messages to.

RestInput
^^^^^^^^^

The ``rest`` channel, will provide you with a REST endpoint to post messages
to and in response to that request will send back the bots messages.
Here is an example on how to connect the ``rest`` input channel
using the run script:

.. code-block:: bash

 rasa core run -m models --port 5002 --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

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

 rasa core run -m models --port 5002 --credentials credentials.yml

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

.. _custom_channels:

Creating a new Channel
----------------------

You can also implement your own, custom channel. You can
use the ``rasa.core.channels.channel.RestInput`` class as a template.
The methods you need to implement are ``blueprint`` and ``name``. The method
needs to create a sanic blueprint that can be attached to a sanic server.

This allows you to add `REST` endpoints to the server, which the external
messaging service can call to deliver messages.

Your blueprint should have at least the two routes ``health`` on ``/``
and ``receive`` on the http route ``/webhook``.

The ``name`` method defines the url prefix. E.g. if your component is
named ``myio`` the webhook you can use to attach the external service is:
``http://localhost:5005/webhooks/myio/webhook`` (replacing the hostname
and port with your values).

To send a message, you would run a command like:

.. code-block:: bash

    curl -XPOST http://localhost:5000/webhooks/myio/webhook \
      -d '{"sender": "user1", "message": "hello"}' \
      -H "Content-type: application/json"

where ``myio`` is the name of your component.

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
channel, please take a look at the implementations of the other
output channels, e.g. the ``SlackBot`` in ``rasa.core.channels.slack``.

To use a custom channel, you need to supply a credentials configuration file
``credentials.yml`` with the command line argument ``--credentials``.
This credentials file has to contain the module path of your custom channel and
any required configuration parameters. This could e.g. look like:

.. code-block:: yaml

    mypackage.MyIO:
      username: "user_name"
      another_parameter: "some value"

Example implementation for an input channel that receives the messages,
hands them over to Rasa Core, collects the bot utterances and returns
these bot utterances as the json response to the webhook call that
posted this message to the channel:

.. literalinclude:: ../../rasa/core/channels/channel.py
   :pyobject: RestInput


.. include:: feedback.inc
