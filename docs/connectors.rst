.. _connectors:

Connecting to messaging & voice platforms
=========================================

Here's how to connect your conversational AI to the outside world.

Input channels are defined in the ``rasa_core.channels`` module.
Currently, there is an implementation for the command line as well as
connection to facebook and slack.

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

Using run script
^^^^^^^^^^^^^^^^
If you want to connect to the slack input channel using the run script, e.g. using

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current \
      --port 5002 --connector slack --credentials slack_credentials.yml

you need to supply a ``slack_credentials.yml`` with the following content:

.. literalinclude:: ../examples/moodbot/slack_credentials.yml
   :linenos:


Directly using python
^^^^^^^^^^^^^^^^^^^^^

A ``SlackInput`` instance provides a flask blueprint for creating
a webserver. This lets you separate the exact endpoints and implementation
from your webserver creation logic.

Code to create a Messenger-compatible webserver looks like this:


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


