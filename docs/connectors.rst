.. _connectors:

Connecting to messaging & voice platforms
=========================================

Here's how to connect your conversational AI to the outside world.

Input channels are defined in the ``rasa_core.channels`` module.
Currently, there is an implementation for the command line as well as
connection to facebook, slack and telegram.

.. _facebook_connector:

Facebook Messenger Setup
------------------------

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to facebook using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector facebook --credentials fb_credentials.yml

you need to supply a ``fb_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/fb_credentials.yml
   :linenos:


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``FacebookInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


.. code-block:: python
    :linenos:

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


Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the slack input channel using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector slack --credentials slack_credentials.yml

Setting Up Webhook
^^^^^^^^^^^^^^^^^^
In order to use this with slack you need a external webhook, typically the best way to do this is use https://ngrok.com/ in order to expose ports externally from your machine.  So in the above example we are using port 5002 so using ngrok we would run:

.. code-block:: bash
  
See ngrok_. for more information on hosting a webhook from your local machine if you don't have a public address or host.

This is what allows slack to send the messages from it to your bot to get the responses.  Once you put in your webhook address in the OAuth & Permissions section and save it you will have the credentials you need for the slack_credentials.yml file.

you need to supply a ``slack_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/slack_credentials.yml
   :linenos:


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``SlackInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a slack-compatible webserver looks like this:


.. code-block:: python
    :linenos:

    from rasa_core.channels import HttpInputChannel
    from rasa_core.channels.slack import SlackInput
    from rasa_core.agent import Agent
    from rasa_core.interpreter import RegexInterpreter

    # load your trained agent
    agent = Agent.load("dialogue", interpreter=RegexInterpreter())

    input_channel = SlackInput(
       slack_token="YOUR_SLACK_TOKEN",  # this is the `bot_user_o_auth_access_token`
       slack_channel="YOUR_SLACK_CHANNEL"  # the name of your channel to which the bot posts
    )

    agent.handle_channel(HttpInputChannel(5004, "/app", input_channel))

The arguments for the ``HttpInputChannel`` are the port, the url prefix, and the input channel.
The default endpoint for receiving facebook messenger messages is ``/webhook``, so the example
above would listen for messages on ``/app/webhook``. This is the url you should add in the
facebook developer portal.


.. _telegram_connector:

Telegram Setup
--------------

Using run script
^^^^^^^^^^^^^^^^

If you want to connect to the slack input channel using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current
      --port 5002 -c telegram --credentials telegram_credentials.yml

you need to supply a ``telegram_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/telegram_credentials.yml
    :linenos:


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``TelegramInput`` instance provides a flask blueprint for creating
a webserver. This lets you seperate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:

.. code-block:: python
    :linenos:

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

      1. To create the bot, go to: https://web.telegram.org/#/im?p=@BotFather, enter
      */newbot* and follow the instructions.
      2. At the end you should get your ``access_token`` and the username you set will
      be your ``verify``.
      3. If you want to use your bot in a group setting, it's advisable to turn on group privacy
      mode by entering */setprivacy*. Then the bot will only listen when the message is started
      with */bot*

    For more information on the Telegram HTTP API, go to https://core.telegram.org/bots/api

.. _ngrok:

Using Ngrok For Local Testing
=========================================
You can use https://ngrok.com/ to create a local webhook from your machine that is Publicly available on the internet so you can use it with applications like Slack, Facebook, etc.

The command to run a ngrok instance for port 5002 for example would be:

.. code-block:: bash
`ngrok httpd 5002`

**Ngrok is only needed if you don't have a public IP and are testing locally**
  
This will then give a output showing a https address that you need to supply for the interactive components request URL and for the incoming webhook and the address should be whatever ngrok supplies you with /webhook added to the end.  This basically takes the code running on your local machine and punches it through the internet at the ngrok address supplied.
