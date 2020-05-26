:desc: Learn more about open-source natural language processing library Rasa NLU a
       for intent classification and entity extraction in on premise chatbots. a
 a
.. _about-rasa-nlu: a
 a
Rasa NLU: Language Understanding for Chatbots and AI assistants a
=============================================================== a
 a
 a
Rasa NLU is an open-source natural language processing tool for intent classification, response retrieval and a
entity extraction in chatbots. For example, taking a sentence like a
 a
.. code-block:: console a
 a
    "I am looking for a Mexican restaurant in the center of town" a
 a
and returning structured data like a
 a
.. code-block:: json a
 a
    { a
      "intent": "search_restaurant", a
      "entities": { a
        "cuisine" : "Mexican", a
        "location" : "center" a
      } a
    } a
 a
If you want to use Rasa NLU on its own, see :ref:`using-nlu-only`. a
 a