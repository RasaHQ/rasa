.. _connectors:

Chat & Voice platforms
======================

Here's how to connect your conversational AI to the outside world.

See :ref:`here <custom_channels>` for details on how to build a custom input channel.

.. contents::

Input channels are defined in the ``rasa_core.channels`` module.
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
credentials file in yaml format, here is an example containing most channels:

.. literalinclude:: ../examples/moodbot/credentials.yml

.. note::

    You only need to include the channels you want to connect to in
    your ``credentials.yml``.


.. _facebook_connector:

Facebook Messenger Setup
------------------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the provided ``rasa_core.run``
script, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the FB credentials:** You need to set up a Facebook app and a page.

  1. To create the app go to: https://developers.facebook.com/
     and click on *"Add a new app"*.
  2. go onto the dashboard for the app and under *Products*,
     click *Add Product* and *add Messenger*. Under the settings for
     Messenger, scroll down to *Token Generation* and click on the
     link to create a new page for your app.
  3. Use the collected ``verify``, ``secret`` and ``access token``
     to connect your bot to facebook.

For more detailed steps, visit the
`messenger docs <https://developers.facebook.com/docs/graph-api/webhooks>`_.


Using the run script
^^^^^^^^^^^^^^^^^^^^
If you want to connect to facebook using the run script, e.g. using:

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector facebook --credentials credentials.yml

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

A ``FacebookInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. literalinclude:: ../tests/test_channels.py
   :pyobject: test_facebook_channel
   :lines: 2-
   :end-before: END DOC INCLUDE

.. _slack_connector:

Slack Setup
-----------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the provided ``rasa_core.run``
script, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Slack credentials:** You need to set up a Slack app.

  1. To create the app go to: https://api.slack.com/apps and click on *"Create New App"*.
  2. Activate the following features: interactive components, event subscriptions, bot users,
     permissions (for basic functionality you should subscribe to the ``message.channel``,
     ``message.groups``, ``message.im`` and ``message.mpim`` events)
  3. The ``slack_channel`` is the target your bot posts to. This can be a channel,
     an app or an individual person
  4. Use the entry for ``Bot User OAuth Access Token`` in the "OAuth & Permissions" tab
     as your ``slack_token``


For more detailed steps, visit the
`slack api docs <https://api.slack.com/incoming-webhooks>`_.

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the slack input channel using the run script, e.g. using:

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector slack --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   slack:
     slack_token: "xoxb-286425452756-safjasdf7sl38KLls"
     slack_channel: "@my_channel"


The endpoint for receiving facebook messenger messages is
``http://localhost:5005/webhooks/slack/webhook``, replacing
the host and port with the appropriate values. This is the URL
you should add in the OAuth & Permissions section.

.. note::

   If you do not set the slack_channel keyword
   argument, messages will by delivered back to
   the user who sent them.

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``SlackInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. literalinclude:: ../tests/test_channels.py
   :pyobject: test_slack_channel
   :lines: 2-
   :end-before: END DOC INCLUDE

.. _mattermost_connector:

Mattermost Setup
----------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the provided ``rasa_core.run``
script, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to setup the outgoing webhook:**

   1. To create the mattermost outgoing webhook login to your mattermost team site and go to **Main Menu > Integrations > Outgoing Webhooks**
   2. Click **Add outgoing webhook**
   3. Fill out the details including the channel you want the bot in.  You will need to ensure the **trigger words** section is setup with @yourbotname so that way it doesn't trigger on everything that is said.
   4. Make sure **trigger when** section is set to value **first word matches a trigger word exactly**
   5. For the callback url this needs to be your ngrok url where you have your webhook running in core or your public address with /webhook example: ``http://test.example.com/webhook``


For more detailed steps, visit the
`mattermost docs at <https://docs.mattermost.com/guides/developer.html>`_.

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the mattermost input channel using the run script, e.g. using:

.. code-block:: bash

 python -m rasa_core.run -d models/dialogue -u models/nlu/current \
     --port 5002 --connector mattermost --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   mattermost:
     url: "https://chat.example.com/api/v4"
     team: "community"
     user: "user@user.com"
     pw: "password"


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``MattermostInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Mattermost-compatible webserver looks like this:

.. literalinclude:: ../tests/test_channels.py
   :pyobject: test_mattermost_channel
   :lines: 2-
   :end-before: END DOC INCLUDE

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving mattermost channel messages is ``/webhook``, so the example
above would listen for messages on ``/app/webhook``. This is the url you should add in the
mattermost outgoing webhook.

.. _telegram_connector:

Telegram Setup
--------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the provided ``rasa_core.run``
script, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Telegram credentials:** You need to set up a Telegram bot.

  1. To create the bot, go to: https://web.telegram.org/#/im?p=@BotFather, enter */newbot* and follow the instructions.
  2. At the end you should get your ``access_token`` and the username you set will be your ``verify``.
  3. If you want to use your bot in a group setting, it's advisable to turn on group privacy mode by entering */setprivacy*. Then the bot will only listen when the message is started with */bot*

For more information on the Telegram HTTP API, go to https://core.telegram.org/bots/api

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to telegram using the run script, e.g. using:

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current
      --port 5002 -c telegram --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   telegram:
     access_token: "490161424:AAGlRxinBRtKGb21_rlOEMtDFZMXBl6EC0o"
     verify: "your_bot"
     webhook_url: "your_url.com/webhook"


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``TelegramInput`` instance provides a flask blueprint for creating
a webserver. This lets you seperate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:

.. literalinclude:: ../tests/test_channels.py
   :pyobject: test_telegram_channel
   :lines: 2-
   :end-before: END DOC INCLUDE

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving messages is ``/webhook``, so the example above above would
listen for messages on ``/app/webhook``. This is the URL you should use as the ``webhook_url``,
so for example ``webhook_url=myurl.com/app/webhook``. To get the bot to listen for messages at
that URL, go to ``myurl.com/app/set_webhook`` first to set the webhook.

.. _twilio_connector:

Twilio Setup
--------------

You first need to retrieve some credentials, once you have them you can
**either** attach the input channel running the provided ``rasa_core.run``
script, or you can attach it in your own code.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Twilio credentials:** You need to set up a Twilio account.

  1. Once you have created a Twilio account, you need to create a new project. The basic important product to select here is ``Programmable SMS``.
  2. Once you have created the project, navigate to the Dashboard of ``Programmable SMS`` and click on ``Get Started`` and follow the steps to connect a phone number to the project.
  3. Now you can use the ``Account SID``, ``Auth Token`` and the phone number you purchased in your credentials yml.

For more information on the Twilio REST API, go to https://www.twilio.com/docs/iam/api

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to the twilio input channel using the run script, e.g. using:

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current
      --port 5002 -c twilio --credentials credentials.yml

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   twillio:
     account_sid: "ACbc2dxxxxxxxxxxxx19d54bdcd6e41186"
     auth_token: "e231c197493a7122d475b4xxxxxxxxxx"
     twilio_number: "+440123456789"

Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``TwilioInput`` instance provides a flask blueprint for creating
a webserver. This lets you seperate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Twilio-compatible webserver looks like this:

.. literalinclude:: ../tests/test_channels.py
   :pyobject: test_twilio_channel
   :lines: 2-
   :end-before: END DOC INCLUDE

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving messages is ``/webhook``, so the example above above would
listen for messages on ``/app/webhook``.

.. _ngrok:

Using Ngrok For Local Testing
-----------------------------

You can use https://ngrok.com/ to create a local webhook from your machine that is Publicly available on the internet so you can use it with applications like Slack, Facebook, etc.

The command to run a ngrok instance for port 5002 for example would be:

.. code-block:: bash

  ngrok httpd 5002

**Ngrok is only needed if you don't have a public IP and are testing locally**

This will then give a output showing a https address that you need to supply for the interactive components request URL and for the incoming webhook and the address should be whatever ngrok supplies you with /webhook added to the end.  This basically takes the code running on your local machine and punches it through the internet at the ngrok address supplied.

.. _custom_channels:

Custom Channels
---------------

If you want to put a widget on your website so that people can talk to your bot, check out these
two projects:

- `Rasa Webchat <https://github.com/mrbot-ai/rasa-webchat>`_ uses sockets.
- `Chatroom <https://github.com/scalableminds/chatroom>`_ uses regular HTTP requests.

You can also implement your own, fully custom channel.


You can also implement your own, custom channel. You can
use the ``rasa_core.channels.channel.RestInput`` class as a template.
The methods you need to implement are ``blueprint`` and ``name``. The method
needs to create a Flask blueprint that can be attached to a flask server.

This allows you to add REST endpoints to the server, the external
messaging service can call to deliver messages.

Your blueprint should have at least the two routes ``health`` on ``/``
and ``receive`` on the http route ``/webhook``.

The ``name`` method defines the url prefix. E.g. if your component is
named ``myio``, the and you start Rasa Core, the
webhook you can use to attach the external service is:
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
is implemented. For examples on how to create and use your own output
channel, please take a look at the implementations of the other
output channels, e.g. the ``SlackBot`` in ``rasa_core.channels.slack``.

To use a custom channel, modify they ``rasa_core.run`` script,
either adding your channel to the
``_create_external_channel`` function or directly overriding the
``input_channel`` variable defined in the
``main`` function.

Example implementation for an input channel that receives the messages,
hands them over to Rasa Core, collects the bot utterances and returns
these bot utterances as the json response to the webhook call that
posted this message to the channel:

.. literalinclude:: ../rasa_core/channels/channel.py
   :pyobject: RestInput
