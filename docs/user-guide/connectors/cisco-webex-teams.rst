:desc: Build a Rasa Chat Bot on Cisco Webex

.. _cisco-webex-teams:

Cisco Webex Teams
=================

You first have to create a cisco webex app to get credentials.
Once you have them you can add these to your ``credentials.yml``.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Cisco Webex Teams credentials:**

You need to set up a bot. Please visit below link to create a bot
`Webex Authentication <https://developer.webex.com/authentication.html>`_.

After you have created the bot through Cisco Webex Teams, you need to create a
room in Cisco Webex Teams. Then add the bot in the room the same way you would
add a person in the room.

You need to note down the room ID for the room you created. This room ID will
be used in ``room`` variable in the ``credentials.yml`` file.

Please follow this link below to find the room ID
``https://developer.webex.com/endpoint-rooms-get.html``

Running on Cisco Webex
^^^^^^^^^^^^^^^^^^^^^^

If you want to connect to the ``webexteams`` input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   webexteams:
     access_token: "YOUR-BOT-ACCESS-TOKEN"
     room: "YOUR-CISCOWEBEXTEAMS-ROOM-ID"


The endpoint for receiving Cisco Webex Teams messages is
``http://localhost:5005/webhooks/webexteams/webhook``, replacing
the host and port with the appropriate values. This is the URL
you should add in the OAuth & Permissions section.

.. note::

   If you do not set the ``room`` keyword
   argument, messages will by delivered back to
   the user who sent them.
