:desc: Learn more about open-source natural language processing library Rasa NLU
       for intent classification and entity extraction in on premise chatbots.

.. _about-rasa-nlu:

Rasa NLU: Language Understanding for Chatbots and AI assistants
===============================================================


Rasa NLU is an open-source natural language processing tool for intent classification and entity extraction in chatbots. For example, taking a sentence like

.. code-block:: console

    "I am looking for a Mexican restaurant in the center of town"

and returning structured data like

.. code-block:: json

    {
      "intent": "search_restaurant",
      "entities": {
        "cuisine" : "Mexican",
        "location" : "center"
      }
    }

Rasa NLU used to be a separate library, but it is now part of the Rasa framework.
