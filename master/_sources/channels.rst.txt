.. _messaging-and-voice-channels:

Messaging and Voice Channels
============================

If you're testing this on your local computer (e.g. not a server), you
will need to `use ngrok </docs/rasa-x/get-feedback-from-test-users/#use-ngrok-for-local-testing>`_ . 
This gives your machine a domain name so that facebook, slack, etc. know where to send messages.


To make your assistant available on a messaging platform you need to provide credentials 
in a ``credentials.yml`` file.
An example file is created when you run ``rasa init``, so it's easiest to edit that file
and add your credentials there. Here is an example with facebook credentials:


.. code-block:: yaml

  facebook:
    verify: "rasa-bot"
    secret: "3e34709d01ea89032asdebfe5a74518"
    page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"


Learn how to make your assistant available on:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   connectors/website
   connectors/messenger
   connectors/slack
   connectors/telegram
   connectors/twilio
   connectors/ms-bot-framework
   connectors/webex
   connectors/rocketchat
   connectors/mattermost
   connectors/custom
