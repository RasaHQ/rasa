:desc: Build a Rasa Chat Bot on Slack

.. _slack:

Slack
=====

.. edit-link::

You first have to create a Slack app to get credentials.
Once you have them you can add these to your ``credentials.yml``.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Slack credentials:** You need to set up a Slack app.

  1. To create the app go to https://api.slack.com/apps and click
     on **Create New App**.
  2. Activate the following features:

    - `Interactive Components <https://api.slack.com/reference/block-kit/interactive-components>`_ (if your bot uses any, e.g. buttons)
    - Event subscriptions
    - Bot users
    - Permissions (for basic functionality you should subscribe to the
      following events: ``message.channel``, ``message.groups``,
      ``message.im`` and ``message.mpim``)

  3. Get the ``Bot User OAuth Access Token`` from the OAuth & Permissions page. You will need
     to provide this value in your credentials later in these instructions. It should start
     with ``xoxb``.


For more detailed steps, visit the
`Slack API docs <https://api.slack.com/incoming-webhooks>`_.

Running on Slack
^^^^^^^^^^^^^^^^

If you want to connect to the slack input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run

Do not forget to run the action server if this is required by your bot,
e.g. using:

.. code-block:: bash

  rasa run actions

You need to supply a ``credentials.yml`` with the following content:

- The ``slack_channel`` is the target your bot posts to.
  This can be a channel or an individual person. You can leave out
  the argument to post DMs to the bot.

- Use the entry for ``Bot User OAuth Access Token`` in the
  "OAuth & Permissions" tab as your ``slack_token``. It should start
  with ``xoxb``.

.. code-block:: yaml

   slack:
     slack_token: "xoxb-286425452756-safjasdf7sl38KLls"
     slack_channel: "#my_channel"
     slack_retry_reason_header: "x-slack-retry-reason"  # Slack HTTP header name indicating reason that slack send retry request. This configuration is optional.
     slack_retry_number_header: "x-slack-retry-num"  # Slack HTTP header name indicating the attempt number. This configuration is optional.
     errors_ignore_retry: None  # Any error codes given by Slack included in this list will be ignored. Error codes are listed `here <https://api.slack.com/events-api#errors>`_.


The endpoint for receiving slack messages is
``http://localhost:5005/webhooks/slack/webhook``, replacing
the host and port with the appropriate values. This is the URL
you should add in the "OAuth & Permissions" section as well as
the "Event Subscriptions".
