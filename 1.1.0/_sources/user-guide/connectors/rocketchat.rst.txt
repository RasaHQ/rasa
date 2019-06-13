:desc: Build a Rasa Chat Bot on Rocketchat

.. _rocketchat:

RocketChat
==========

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to set up Rocket.Chat:**

 1. Create a user that will be used to post messages, and set its
    credentials at credentials file.
 2. Create a Rocket.Chat outgoing webhook by logging in as admin to
    Rocket.Chat and going to
    **Administration > Integrations > New Integration**.
 3. Select **Outgoing Webhook**.
 4. Set **Event Trigger** section to value **Message Sent**.
 5. Fill out the details, including the channel you want the bot
    listen to. Optionally, it is possible to set the
    **Trigger Words** section with ``@yourbotname`` so that the bot
    doesn't trigger on everything that is said.
 6. Set your **URLs** section to the Rasa URL where you have your
    webhook running in Core or your public address with
    ``/webhooks/rocketchat/webhook``, e.g.
    ``http://test.example.com/webhooks/rocketchat/webhook``.

For more information on the Rocket.Chat Webhooks, see the
`Rocket.Chat Guide <https://rocket.chat/docs/administrator-guides/integrations/>`_.


Running on RocketChat
^^^^^^^^^^^^^^^^^^^^^

If you want to connect to the Rocket.Chat input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   rocketchat:
     user: "yourbotname"
     password: "YOUR_PASSWORD"
     server_url: "https://demo.rocket.chat"
