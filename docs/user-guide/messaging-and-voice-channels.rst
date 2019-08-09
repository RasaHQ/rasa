:desc: Check out how to make your Rasa assistant available on platforms like
       Facebook Messenger, Slack, Telegram or even your very own website. 

.. _messaging-and-voice-channels:

Messaging and Voice Channels
============================

.. edit-link::

If you're testing this on your local computer (i.e. not a server), you
will need to use `ngrok <https://rasa.com/docs/rasa-x/get-feedback-from-test-users/#use-ngrok-for-local-testing>`_.
This gives your machine a domain name so that Facebook, Slack, etc. know where to send messages to
reach your local machine.


To make your assistant available on a messaging platform you need to provide credentials
in a ``credentials.yml`` file.
An example file is created when you run ``rasa init``, so it's easiest to edit that file
and add your credentials there. Here is an example with Facebook credentials:


.. code-block:: yaml

  facebook:
    verify: "rasa-bot"
    secret: "3e34709d01ea89032asdebfe5a74518"
    page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"


Learn how to make your assistant available on:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   connectors/your-own-website
   connectors/facebook-messenger
   connectors/slack
   connectors/telegram
   connectors/twilio
   connectors/microsoft-bot-framework
   connectors/cisco-webex-teams
   connectors/rocketchat
   connectors/mattermost
   connectors/custom-connectors
