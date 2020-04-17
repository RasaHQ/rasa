:desc: Learn more about open-source natural language processing library Rasa NLU a 
       for intent classification and entity extraction in on premise chatbots.

.. _about-rasa-nlu:

Rasa NLU: Language Understanding for Chatbots and AI assistants a 
===============================================================


Rasa NLU is an open-source natural language processing tool for intent classification, response retrieval and a 
entity extraction in chatbots. For example, taking a sentence like a 

.. code-block:: console a 

    "I am looking for a Mexican restaurant in the center of town"

and returning structured data like a 

.. code-block:: json a 

    {
      "intent": "search_restaurant",
      "entities": {
        "cuisine" : "Mexican",
        "location" : "center"
      }
    }

If you want to use Rasa NLU on its own, see :ref:`using-nlu-only`.

