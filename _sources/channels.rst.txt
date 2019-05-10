.. _messaging-and-voice-channels:

Messaging and Voice Channels
============================

If you're testing this on your local computer (e.g. not a server), you
will need to :ref:`use_ngrok_for_local_testing`. 
This gives your machine a domain name and so that facebook, slack, etc. know where to send messages.


Make your assistant available to users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To make your assistant available on a messaging platform like :ref:`slack-connector` or
:ref:`messenger-connector`, you need to provide credentials in a ``credentials.yml`` file.
An example file is created when you run ``rasa init``, so it's easiest to edit that file
and add your credentials there.

Here is an example file containing facebook credentials:


.. code-block:: yaml

  facebook:
    verify: "rasa-bot"
    secret: "3e34709d01ea89032asdebfe5a74518"
    page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"


Here is the full list of connectors:

.. toctree::
   :titlesonly:
   :maxdepth: 1

   website
   messenger
   slack
   telegram
   twilio
   ms-bot-framework
   webex
   rocketchat
   mattermost
   custom
