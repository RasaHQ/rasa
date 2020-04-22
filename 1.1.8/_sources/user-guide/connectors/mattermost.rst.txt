:desc: Build a Rasa Chat Bot on Mattermost

.. _mattermost:

Mattermost
----------

You first have to create a mattermost app to get credentials.
Once you have them you can add these to your ``credentials.yml``.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to set up the outgoing webhook:**

   1. To create the Mattermost outgoing webhook, login to your Mattermost
      team site and go to **Main Menu > Integrations > Outgoing Webhooks**.
   2. Click **Add outgoing webhook**.
   3. Fill out the details including the channel you want the bot in.
      You will need to ensure the **trigger words** section is set up
      with ``@yourbotname`` so that the bot doesn't trigger on everything
      that is said.
   4. Make sure **trigger when** is set to value
      **first word matches a trigger word exactly**.
   5. The callback url needs to be your ngrok url where you
      have your webhook running in Core or your public address, e.g.
      ``http://test.example.com/webhooks/mattermost/webhook``.


For more detailed steps, visit the
`Mattermost docs <https://docs.mattermost.com/guides/developer.html>`_.

Running on Mattermost
^^^^^^^^^^^^^^^^^^^^^

If you want to connect to the Mattermost input channel using the
run script, e.g. using:

.. code-block:: bash

   rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   mattermost:
     url: "https://chat.example.com/api/v4"
     team: "community"
     user: "user@user.com"
     pw: "password"

The endpoint for receiving Mattermost channel messages
is ``/webhooks/mattermost/webhook``. This is the url you should
add in the Mattermost outgoing webhook.
