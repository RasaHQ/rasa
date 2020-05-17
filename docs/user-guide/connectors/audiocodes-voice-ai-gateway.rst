:desc: Build a Rasa Chat Bot on AudioCodes Voice.AI Gateway

.. _audiocodes:

AudioCodes
==========

.. edit-link::

AudioCodes Voice-AI Gateway is an application that enables telephony access
for chatbots.

The Rasa integration is using a REST protocol named AC-Bot-API.

Setting Credentials
^^^^^^^^^^^^^^^^^^^

In order to integrate with AudioCodes Voice.AI Gateway, configure
a provider on VAIG side with the following attributes:

.. code-block:: json

  {
    "name": "rasa",
    "type": "ac-bot-api",
    "botURL": "https://<RASA>/webhooks/audiocodes/webhook",
    "credentials": {
      "token": "CHOOSE_YOUR_TOKEN"
    }
  }

Running on AudioCodes Voice.AI Gateway
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to connect to the AudioCodes input channel using the run
script, e.g. using:

.. code-block:: bash

  rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

  audiocodes:
    token: "CHOOSE_YOUR_TOKEN"
