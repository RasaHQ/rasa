:desc: Read more how to define assistant utterances or use a service to generate the
       responses using Rasa Stack as an open source chatassistant platform.

.. _responses:

Responses
=========

If you want your assistant to respond to user messages, you need to manage
these responses. In the training data for your bot,
your stories, you specify the actions your bot
should execute. These actions
can use utterances to send messages back to the user.

There are two ways to manage these utterances:

1. Include your assistant utterances in your domain file, or
2. Use an external service to generate the responses.

Including the utterances in the domain
--------------------------------------

The default format is to include the utterances in your domain file.
This file then contains references to all your custom actions,
available entities, slots and intents.

.. literalinclude:: ../../data/test_domains/default_with_slots.yml
   :language: yaml

In this example domain file, the section ``templates`` contains the
template the assistant uses to send messages to the user.

If you want to change the text, or any other part of the bots response,
you need to retrain the assistant before these changes will be picked up.

More details about the format of these responses can be found in the
documentation about the domain file format: :ref:`utter_templates`.


Managing assistant utterances using an external CMS
---------------------------------------------------

Retraining the bot just to change the text copy can be suboptimal for
some workflows. That's why Core also allows you to outsource the
response generation and separate it from the dialogue learning.

The assistant will still learn to predict actions and to react to user input
based on past dialogues, but the responses it sends back to the user
are generated outside of Rasa Core.

If the assistant wants to send a message to the user, it will call an
external HTTP server with a ``POST`` request. To configure this endpoint,
you need to create an ``endpoints.yml`` and pass it either to the ``run``
or ``server`` script. The content of the ``endpoints.yml`` should be

.. literalinclude:: ../../data/test_endpoints/example_endpoints.yml
   :language: yaml

Then pass the ``enable-api`` flag to the ``rasa run`` command when starting
the server:

.. code-block:: shell

    $ rasa run \
       --enable-api \
       -m examples/babi/models \
       --log-file out.log \
       --endpoints endpoints.yml


The body of the ``POST`` request sent to the endpoint will look
like this:

.. code-block:: json

  {
    "tracker": {
      "latest_message": {
        "text": "/greet",
        "intent_ranking": [
          {
            "confidence": 1.0,
            "name": "greet"
          }
        ],
        "intent": {
          "confidence": 1.0,
          "name": "greet"
        },
        "entities": []
      },
      "sender_id": "22ae96a6-85cd-11e8-b1c3-f40f241f6547",
      "paused": false,
      "latest_event_time": 1531397673.293572,
      "slots": {
        "name": null
      },
      "events": [
        {
          "timestamp": 1531397673.291998,
          "event": "action",
          "name": "action_listen"
        },
        {
          "timestamp": 1531397673.293572,
          "parse_data": {
            "text": "/greet",
            "intent_ranking": [
              {
                "confidence": 1.0,
                "name": "greet"
              }
            ],
            "intent": {
              "confidence": 1.0,
              "name": "greet"
            },
            "entities": []
          },
          "event": "user",
          "text": "/greet"
        }
      ]
    },
    "arguments": {},
    "template": "utter_greet",
    "channel": {
      "name": "collector"
    }
  }

The endpoint then needs to respond with the generated response:

.. code-block:: json

  {
      "text": "hey there",
      "buttons": [],
      "image": null,
      "elements": [],
      "attachments": []
  }

Rasa will then use this response and sent it back to the user.
