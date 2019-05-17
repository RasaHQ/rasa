:desc: Build a Rasa Chat Bot on Telegram

.. _telegram_connector:

Telegram
========

You first have to create a telegram bot to get credentials.
Once you have them you can add these to your ``credentials.yml``.

Getting Credentials
^^^^^^^^^^^^^^^^^^^

**How to get the Telegram credentials:**
You need to set up a Telegram bot.

  1. To create the bot, go to: https://web.telegram.org/#/im?p=@BotFather,
     enter */newbot* and follow the instructions.
  2. At the end you should get your ``access_token`` and the username you
     set will be your ``verify``.
  3. If you want to use your bot in a group setting, it's advisable to
     turn on group privacy mode by entering */setprivacy*. Then the bot
     will only listen when the message is started with */bot*

For more information on the Telegram HTTP API, go to
https://core.telegram.org/bots/api

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
     webhook_url: "your_url.com/webhooks/telegram/webhook"
