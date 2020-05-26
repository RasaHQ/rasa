:desc: Check out how to make your Rasa assistant available on platforms like a
       Facebook Messenger, Slack, Telegram or even your very own website.  a
 a
.. _messaging-and-voice-channels: a
 a
Messaging and Voice Channels a
============================ a
 a
.. edit-link:: a
 a
To make your assistant available on a messaging platform you need to provide credentials a
in a ``credentials.yml`` file. a
An example file is created when you run ``rasa init``, so it's easiest to edit that file a
and add your credentials there. Here is an example with Facebook credentials: a
 a
 a
.. code-block:: yaml a
 a
  facebook: a
    verify: "rasa-bot" a
    secret: "3e34709d01ea89032asdebfe5a74518" a
    page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD" a
 a
 a
Learn how to make your assistant available on: a
 a
.. toctree:: a
   :titlesonly: a
   :maxdepth: 1 a
 a
   connectors/your-own-website a
   connectors/facebook-messenger a
   connectors/slack a
   connectors/telegram a
   connectors/twilio a
   connectors/microsoft-bot-framework a
   connectors/cisco-webex-teams a
   connectors/rocketchat a
   connectors/mattermost a
   connectors/hangouts a
   connectors/custom-connectors a
 a
 a
.. _using-ngrok: a
 a
Testing Channels on Your Local Machine with Ngrok a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
You can use `ngrok <https://ngrok.com/>`_ to create a connection to your local a
computer that is publicly available on the internet. a
You don't need this when running Rasa on a server because, you can set up a domain a
name to point to that server's IP address, or use the IP address itself. a
 a
After installing ngrok, run: a
 a
.. copyable:: a
 a
   ngrok http 5005; rasa run a
 a
Your webhook address will look like the following: a
 a
- ``https://yyyyyy.ngrok.io/webhooks/<CHANNEL>/webhook``, e.g. a
- ``https://yyyyyy.ngrok.io/webhooks/facebook/webhook`` a
 a
.. warning:: a
 a
  With the free-tier of ngrok, you can run into limits on how many connections you can make per minute. a
  As of writing this, it is set to 40 connections / minute. a
 a
 a