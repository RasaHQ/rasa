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

Supported Response Attachments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to standard ``text:`` responses, this channel also supports the following components from the `Telegram API <https://core.telegram.org/bots/api/#message>`_:

- ``button`` arguments:
  - button_type: inline | vertical | reply
- ``custom`` arguments: 
  - photo
  - audio
  - document
  - sticker
  - video
  - video_note
  - animation
  - voice
  - media
  - latitude, longitude (location)
  - latitude, longitude, title, address (venue)
  - phone_number
  - game_short_name
  - action

Examples:

.. code-block:: yaml

  utter_ask_transfer_form_confirm:
  - buttons:
    - payload: /affirm
      title: Yes
    - payload: /deny
      title: No, cancel the transaction
    button_type: vertical
    text: Would you like to transfer {currency}{amount_of_money} to {PERSON}?
    image: "https://i.imgur.com/nGF1K8f.jpg"

.. code-block:: yaml

  - text: Here's my giraffe sticker!
    custom:
      sticker: "https://github.com/TelegramBots/book/raw/master/src/docs/sticker-fred.webp"
