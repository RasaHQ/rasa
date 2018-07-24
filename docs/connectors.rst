.. _connectors:

Chat & Voice platforms
======================

Here's how to connect your conversational AI to the outside world.

.. contents::

Input channels are defined in the ``rasa_core.channels`` module.
Currently, there is code for connecting to
facebook, slack, telegram, mattermost and twilio. If the connection
you want is missing, this is a great place to start contributing!

If you're testing on your local machine (e.g. not a server), you 
will need to use ngrok_. This gives your machine a domain name
and so that facebook, slack, etc. know where to send messages.

.. _facebook_connector:

Facebook Messenger Setup
------------------------

Using the run script
^^^^^^^^^^^^^^^^^^^^
If you want to connect to facebook using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector facebook --credentials fb_credentials.yml

you need to supply a ``fb_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/fb_credentials.yml


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``FacebookInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. code-block:: python

    from rasa_core.channels import HttpInputChannel
    from rasa_core.channels.facebook import FacebookInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load("dialogue", interpreter=RegexInterpreter())

    input_channel = FacebookInput(
       fb_verify="YOUR_FB_VERIFY",  # you need tell facebook this token, to confirm your URL
       fb_secret="YOUR_FB_SECRET",  # your app secret
       fb_access_token="YOUR_FB_PAGE_ACCESS_TOKEN"   # token for the page you subscribed to
    )

    agent.handle_channel(HttpInputChannel(5004, "/app", input_channel))

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving facebook messenger messages is ``/webhook``, so the example
above would listen for messages on ``/app/webhook``. This is the url you should add in the
facebook developer portal.

.. note::

   **How to get the FB credentials:** You need to set up a Facebook app and a page.

      1. To create the app go to: https://developers.facebook.com/ and click on *"Add a new app"*.
      2. go onto the dashboard for the app and under *Products*, click *Add Product* and *add Messenger*. Under the settings for Messenger, scroll down to *Token Generation* and click on the link to create a new page for your app.
      3. Use the collected ``verify``, ``secret`` and ``access token`` to connect your bot to facebook.

   For more detailed steps, visit the
   `messenger docs <https://developers.facebook.com/docs/graph-api/webhooks>`_.


.. _slack_connector:

Slack Setup
-----------

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the slack input channel using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector slack --credentials slack_credentials.yml

you need to supply a ``slack_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/slack_credentials.yml


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``SlackInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. code-block:: python

    from rasa_core.channels import HttpInputChannel
    from rasa_core.channels.slack import SlackInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load("dialogue", interpreter=RegexInterpreter())

    input_channel = SlackInput(
       slack_token="YOUR_SLACK_TOKEN",  # this is the `bot_user_o_auth_access_token`
       slack_channel="YOUR_SLACK_CHANNEL"  # the name of your channel to which the bot posts (optional)
    )

    agent.handle_channel(HttpInputChannel(5004, "/app", input_channel))

The arguments for the HttpInputChannel are the port, the url prefix, and the input channel.
The default endpoint for receiving messages is /webhook, so the example above would listen for messages on /app/webhook.
This is the url you should add in the OAuth & Permissions section. N.b. if you do not set the slack_channel keyword
argument, messages will by delivered back to the user who sent them.

.. note::

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


.. _mattermost_connector:

Mattermost Setup
----------------

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the mattermost input channel using the run script, e.g. using

.. code-block:: bash

 python -m rasa_core.run -d models/dialogue -u models/nlu/current \
     --port 5002 --connector mattermost --credentials mattermost_credentials.yml

you need to supply a ``mattermost_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/mattermost_credentials.yml


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``MattermostInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Mattermost-compatible webserver looks like this:


.. code-block:: python

   from rasa_core.channels import HttpInputChannel
   from rasa_core.channels.slack import MattermostInput
   from rasa_core.agent import Agent
   from rasa_core.interpreter import RegexInterpreter

   # load your trained agent
   agent = Agent.load("dialogue", interpreter=RegexInterpreter())

   input_channel = MattermostInput(
      url="http://chat.example.com/api/v4",  # this is the url of the api for your mattermost instance
      team="community"  # the name of your team for mattermost
      user="user@email.com" # the username of your bot user that will post messages
      pw="password" # the password of your bot user that will post messages
   )

   agent.handle_channel(HttpInputChannel(5004, "/app", input_channel))

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving mattermost channel messages is ``/webhook``, so the example
above would listen for messages on ``/app/webhook``. This is the url you should add in the
mattermost outgoing webhook.

.. note::

  **How to setup the outgoing webhook:**

     1. To create the mattermost outgoing webhook login to your mattermost team site and go to **Main Menu > Integrations > Outgoing Webhooks**
     2. Click **Add outgoing webhook**
     3. Fill out the details including the channel you want the bot in.  You will need to ensure the **trigger words** section is setup with @yourbotname so that way it doesn't trigger on everything that is said.
     4. Make sure **trigger when** section is set to value **first word matches a trigger word exactly**
     5. For the callback url this needs to be your ngrok url where you have your webhook running in core or your public address with /webhook example: ``http://test.example.com/webhook``


  For more detailed steps, visit the
  `mattermost docs at <https://docs.mattermost.com/guides/developer.html>`_.

.. _telegram_connector:

Telegram Setup
--------------

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to telegram using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current
      --port 5002 -c telegram --credentials telegram_credentials.yml

you need to supply a ``telegram_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/telegram_credentials.yml


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``TelegramInput`` instance provides a flask blueprint for creating
a webserver. This lets you seperate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:

.. code-block:: python

    from rasa_core.channels import HttpInputChannel
    from rasa_core.channels.telegram import TelegramInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load("dialogue", interpreter=RegexInterpreter())

    input_channel = TelegramInput(
      access_token="YOUR_ACCESS_TOKEN", # you get this when setting up a bot
      verify="YOUR_TELEGRAM_BOT", # this is your bots username
      webhook_url="YOUR_WEBHOOK_URL" # the url your bot should listen for messages
    )

    agent.handle_channel(HttpInputChannel(5004, "/app", input_channel))

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving messages is ``/webhook``, so the example above above would
listen for messages on ``/app/webhook``. This is the URL you should use as the ``webhook_url``,
so for example ``webhook_url=myurl.com/app/webhook``. To get the bot to listen for messages at
that URL, go to ``myurl.com/app/set_webhook`` first to set the webhook.

.. note::

    **How to get the Telegram credentials:** You need to set up a Telegram bot.

      1. To create the bot, go to: https://web.telegram.org/#/im?p=@BotFather, enter */newbot* and follow the instructions.
      2. At the end you should get your ``access_token`` and the username you set will be your ``verify``.
      3. If you want to use your bot in a group setting, it's advisable to turn on group privacy mode by entering */setprivacy*. Then the bot will only listen when the message is started with */bot*

    For more information on the Telegram HTTP API, go to https://core.telegram.org/bots/api

.. _twilio_connector:

Twilio Setup
--------------

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to the twilio input channel using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current
      --port 5002 -c twilio --credentials twilio_credentials.yml

you need to supply a ``twilio_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/twilio_credentials.yml


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``TwilioInput`` instance provides a flask blueprint for creating
a webserver. This lets you seperate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Twilio-compatible webserver looks like this:

.. code-block:: python

    from rasa_core.channels import HttpInputChannel
    from rasa_core.channels.twilio import TwilioInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load("dialogue", interpreter=RegexInterpreter())

    input_channel = TwilioInput(
      account_sid="YOUR_ACCOUNT_SID", # you get this from your twilio account
      auth_token="YOUR_AUTH_TOKEN", # also from your twilio account
      twilio_number="YOUR_TWILIO_NUMBER" # a number associated with your twilio account
    )

    agent.handle_channel(HttpInputChannel(5004, "/app", input_channel))

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving messages is ``/webhook``, so the example above above would
listen for messages on ``/app/webhook``.

.. note::

    **How to get the Twilio credentials:** You need to set up a Twilio account.

      1. Once you have created a Twilio account, you need to create a new project. The basic important product to select here is ``Programmable SMS``.
      2. Once you have created the project, navigate to the Dashboard of ``Programmable SMS`` and click on ``Get Started`` and follow the steps to connect a phone number to the project.
      3. Now you can use the ``Account SID``, ``Auth Token`` and the phone number you purchased in your credentials yml.

    For more information on the Twilio REST API, go to https://www.twilio.com/docs/iam/api

.. _ngrok:

Using Ngrok For Local Testing
-----------------------------

You can use https://ngrok.com/ to create a local webhook from your machine that is Publicly available on the internet so you can use it with applications like Slack, Facebook, etc.

The command to run a ngrok instance for port 5002 for example would be:

.. code-block:: bash

  ngrok httpd 5002

**Ngrok is only needed if you don't have a public IP and are testing locally**

This will then give a output showing a https address that you need to supply for the interactive components request URL and for the incoming webhook and the address should be whatever ngrok supplies you with /webhook added to the end.  This basically takes the code running on your local machine and punches it through the internet at the ngrok address supplied.


Custom Channels
---------------

You can also implement your own, custom channel. 
You can use the ``rasa_core.channels.custom.CustomInput`` class as a template.
The most important methods to define are ``CustomInput.receive`` and ``CustomOutput.send_text_message``.
These specify how Rasa Core will read incoming messages form the requests sent by your app, and the format
to use when sending messages back to the user. 

To use a custom channel, modify they ``rasa_core.run`` script, either adding your channel to the 
``_create_external_channel`` function or directly overriding the ``input_channel`` variable defined in the 
``main`` function.

To define a custom channel, you need to define a HTTP endpoint that will receive messages. 

.. literalinclude:: ../rasa_core/channels/custom.py 
   :pyobject: CustomInput

To send a message, you would run a command like:

.. code-block:: bash

    curl -XPOST http://localhost:5000/webhook \
      -d '{"sender": "user1", "message": "hello"}' \
      -H "Content-type: application/json"

To get responses, you also need an output channel which implements sending messages back to the user.

.. literalinclude:: ../rasa_core/channels/custom.py 
   :pyobject: CustomOutput
