:desc: Build a Rasa Chat Bot on Telegram

.. _telegram:

Telegram
========

.. edit-link::

You first have to create a Telegram bot to get credentials.
Once you have them you can add these to your ``credentials.yml``.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Telegram credentials:**
You need to set up a Telegram bot.

  1. To create the bot, go to `Bot Father <https://web.telegram.org/#/im?p=@BotFather>`_,
     enter ``/newbot`` and follow the instructions.
  2. At the end you should get your ``access_token`` and the username you
     set will be your ``verify``.
  3. If you want to use your bot in a group setting, it's advisable to
     turn on group privacy mode by entering ``/setprivacy``. Then the bot
     will only listen when a user's message starts with ``/bot``.

For more information, check out the `Telegram HTTP API
<https://core.telegram.org/bots/api>`_.

Running on Telegram
^^^^^^^^^^^^^^^^^^^

If you want to connect to telegram using the run script, e.g. using:

.. code-block:: bash

  rasa run

you need to supply a ``credentials.yml`` with the following content:

.. code-block:: yaml

   telegram:
     access_token: "490161424:AAGlRxinBRtKGb21_rlOEMtDFZMXBl6EC0o"
     verify: "your_bot"
     webhook_url: "https://your_url.com/webhooks/telegram/webhook"
