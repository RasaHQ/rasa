:desc: Read how to define assistant responses or use a service to generate the a
       responses using Rasa as an open source chat assistant platform. a
 a
.. _responses: a
 a
Responses a
========= a
 a
.. edit-link:: a
 a
If you want your assistant to respond to user messages, you need to manage a
these responses. In the training data for your bot, a
your stories, you specify the actions your bot a
should execute. These actions a
can use responses to send messages back to the user. a
 a
There are three ways to manage these responses: a
 a
1. Responses are normally stored in your domain file, see :ref:`here <domain-responses>` a
2. Retrieval action responses are part of the training data, see :ref:`here <retrieval-actions>` a
3. You can also create a custom NLG service to generate responses, see :ref:`here <custom-nlg-service>` a
 a
.. _in-domain-responses: a
 a
Including the responses in the domain a
-------------------------------------- a
 a
The default format is to include the responses in your domain file. a
This file then contains references to all your custom actions, a
available entities, slots and intents. a
 a
.. literalinclude:: ../../data/test_domains/default_with_slots.yml a
   :language: yaml a
 a
In this example domain file, the section ``responses`` contains the a
responses the assistant uses to send messages to the user. a
 a
.. note:: a
 a
    If you want to change the text, or any other part of the bots response, a
    you need to retrain the assistant before these changes will be picked up. a
 a
.. note:: a
 a
    Responses that are used in a story should be listed in the ``stories`` a
    section of the domain.yml file. In this example, the ``utter_channel`` a
    response is not used in a story so it is not listed in that section. a
 a
More details about the format of these responses can be found in the a
documentation about the domain file format: :ref:`domain-responses`. a
 a
.. _custom-nlg-service: a
 a
Creating your own NLG service for bot responses a
----------------------------------------------- a
 a
Retraining the bot just to change the text copy can be suboptimal for a
some workflows. That's why Core also allows you to outsource the a
response generation and separate it from the dialogue learning. a
 a
The assistant will still learn to predict actions and to react to user input a
based on past dialogues, but the responses it sends back to the user a
are generated outside of Rasa Core. a
 a
If the assistant wants to send a message to the user, it will call an a
external HTTP server with a ``POST`` request. To configure this endpoint, a
you need to create an ``endpoints.yml`` and pass it either to the ``run`` a
or ``server`` script. The content of the ``endpoints.yml`` should be a
 a
.. literalinclude:: ../../data/test_endpoints/example_endpoints.yml a
   :language: yaml a
 a
Then pass the ``enable-api`` flag to the ``rasa run`` command when starting a
the server: a
 a
.. code-block:: shell a
 a
    $ rasa run \ a
       --enable-api \ a
       -m examples/babi/models \ a
       --log-file out.log \ a
       --endpoints endpoints.yml a
 a
 a
The body of the ``POST`` request sent to the endpoint will look a
like this: a
 a
.. code-block:: json a
 a
  { a
    "tracker": { a
      "latest_message": { a
        "text": "/greet", a
        "intent_ranking": [ a
          { a
            "confidence": 1.0, a
            "name": "greet" a
          } a
        ], a
        "intent": { a
          "confidence": 1.0, a
          "name": "greet" a
        }, a
        "entities": [] a
      }, a
      "sender_id": "22ae96a6-85cd-11e8-b1c3-f40f241f6547", a
      "paused": false, a
      "latest_event_time": 1531397673.293572, a
      "slots": { a
        "name": null a
      }, a
      "events": [ a
        { a
          "timestamp": 1531397673.291998, a
          "event": "action", a
          "name": "action_listen" a
        }, a
        { a
          "timestamp": 1531397673.293572, a
          "parse_data": { a
            "text": "/greet", a
            "intent_ranking": [ a
              { a
                "confidence": 1.0, a
                "name": "greet" a
              } a
            ], a
            "intent": { a
              "confidence": 1.0, a
              "name": "greet" a
            }, a
            "entities": [] a
          }, a
          "event": "user", a
          "text": "/greet" a
        } a
      ] a
    }, a
    "arguments": {}, a
    "template": "utter_greet", a
    "channel": { a
      "name": "collector" a
    } a
  } a
 a
The endpoint then needs to respond with the generated response: a
 a
.. code-block:: json a
 a
  { a
      "text": "hey there", a
      "buttons": [], a
      "image": null, a
      "elements": [], a
      "attachments": [] a
  } a
 a
Rasa will then use this response and sent it back to the user. a
 a
 a
.. _external-events: a
 a
Proactively Reaching Out to the User with External Events a
--------------------------------------------------------- a
 a
You may want to proactively reach out to the user, a
for example to display the output of a long running background operation a
or notify the user of an external event. a
To learn more, check out `reminderbot <https://github.com/RasaHQ/rasa/blob/master/examples/reminderbot/README.md>`_ in a
the Rasa examples directory or look into :ref:`reminders-and-external-events`. a
 a