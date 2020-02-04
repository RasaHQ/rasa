:desc: Check out how to make your Rasa assistant available on platforms like
       Facebook Messenger, Slack, Telegram or even your very own website. 

.. _messaging-and-voice-channels:

Messaging and Voice Channels
============================

.. edit-link::

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


.. _using-ngrok:

Testing Channels on Your Local Machine with Ngrok
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use `ngrok <https://ngrok.com/>`_ to create a connection to your local
computer that is publicly available on the internet.
You don't need this when running Rasa on a server because, you can set up a domain
name to point to that server's IP address, or use the IP address itself.

After installing ngrok, run:

.. copyable::

   ngrok http 5005; rasa run

Your webhook address will look like the following:

- ``https://yyyyyy.ngrok.io/webhooks/<CHANNEL>/webhook``, e.g.
- ``https://yyyyyy.ngrok.io/webhooks/facebook/webhook``

.. warning::

  With the free-tier of ngrok, you can run into limits on how many connections you can make per minute.
  As of writing this, it is set to 40 connections / minute.

