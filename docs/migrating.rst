.. _section_migration:

Migrating an existing app
=========================

Rasa NLU is designed to make migrating from wit/LUIS/Dialogflow as simple as possible.
The TLDR instructions for migrating are: 

- download an export of your app data from wit/LUIS/Dialogflow
- follow the :ref:`tutorial`, using your downloaded data instead of ``demo-rasa.json``


Banana Peels
------------

Just some specific things to watch out for for each of the services you might want to migrate from

wit.ai
^^^^^^

Wit used to handle ``intents`` natively. 
Now they are somewhat obfuscated. 
To create an ``intent`` in wit you have to create and ``entity`` which spans the entire text.
The file you want from your download is called ``expressions.json``

LUIS.ai
^^^^^^^

Nothing special here. Downloading the data and importing it into Rasa NLU should work without issues

Dialogflow
^^^^^^^^^^

Dialogflow exports generate multiple files rather than just one.
Put them all in a directory (see ``data/examples/dialogflow`` in the repo)
and pass that path to the trainer. 



Emulation
---------

To make Rasa NLU easy to try out with existing projects, the server can `emulate` wit, LUIS, or Dialogflow.
In native mode, a request / response looks like this : 

.. code-block:: console

    $ curl -XPOST localhost:5000/parse -d '{"q":"I am looking for Chinese food"}' | python -mjson.tool
    {
      "text": "I am looking for Chinese food", 
      "intent": "restaurant_search", 
      "confidence": 0.4794813722432127,
      "entities": [
        {
          "start": 17,
          "end": 24, 
          "value": "chinese", 
          "entity": "cuisine"
        }
      ]
    }


if we run in ``wit`` mode (e.g. ``python -m rasa_nlu.server -e wit``)

then instead have to make a GET request

.. code-block:: console

    $ curl 'localhost:5000/parse?q=hello' | python -mjson.tool
    [
        {
            "_text": "hello",
            "confidence": 0.4794813722432127,
            "entities": {},
            "intent": "greet"
        }
    ]

similarly for LUIS, but with a slightly different response format


.. code-block:: console

    $ curl 'localhost:5000/parse?q=hello' | python -mjson.tool
    {
        "entities": [],
        "query": "hello",
        "topScoringIntent": {
            "intent": "inform",
            "score": 0.4794813722432127
        }
    }

and finally for Dialogflow

.. code-block:: console

    $ curl 'localhost:5000/parse?q=hello' | python -mjson.tool
    {
        "id": "ffd7ede3-b62f-11e6-b292-98fe944ee8c2",
        "result": {
            "action": null,
            "actionIncomplete": null,
            "contexts": [],
            "fulfillment": {},
            "metadata": {
                "intentId": "ffdbd6f3-b62f-11e6-8504-98fe944ee8c2",
                "intentName": "greet",
                "webhookUsed": "false"
            },
            "parameters": {},
            "resolvedQuery": "hello",
            "score": null,
            "source": "agent"
        },
        "sessionId": "ffdbd814-b62f-11e6-93b2-98fe944ee8c2",
        "status": {
            "code": 200,
            "errorType": "success"
        },
        "timestamp": "2016-11-29T12:33:15.369411"
    }