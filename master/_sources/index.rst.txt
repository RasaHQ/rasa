
Language Understanding with Rasa NLU
====================================

.. note::
    This is the documentation for version |release| of Rasa NLU. Make sure you select
    the appropriate version of the documentation for your local installation!


Rasa NLU is an open-source tool for intent classification and entity extraction. For example, taking a sentence like

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


The intended audience is mainly people developing bots. 
You can use Rasa as a drop-in replacement for `wit <https://wit.ai>`_ , `LUIS <https://www.luis.ai>`_ , or `Dialogflow <https://dialogflow.com>`_, the only change in your code is to send requests to ``localhost`` instead (see :ref:`section_migration` for details).

Why might you use Rasa instead of one of those services?

- you don't have to hand over your data to FB/MSFT/GOOG
- you don't have to make a ``https`` call every time.
- you can tune models to work well on your particular use case.

These points are laid out in more detail in a `blog post <https://medium.com/lastmile-conversations/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d>`_ .


The quickest quickstart in the west
-----------------------------------


.. code-block:: console

    $ python setup.py install
    $ python -m rasa_nlu.server -e wit &
    $ curl 'http://localhost:5000/parse?q=hello'
    [{"_text": "hello", "confidence": 1.0, "entities": {}, "intent": "greet"}]


There you go! you just parsed some text. Next step, do the :ref:`section_tutorial`.

.. note:: This demo uses a very limited ML model. To apply Rasa NLU to your use case, you need to train your own model! Follow the tutorial to get to know how to apply rasa_nlu to your data.

About 
-----

You can think of Rasa NLU as a set of high level APIs for building your own language parser using existing NLP and ML libraries.
The setup process is designed to be as simple as possible. If you're currently using wit, LUIS, or Dialogflow, you just:

1. download your app data from wit or LUIS and feed it into Rasa NLU
2. run Rasa NLU on your machine and switch the URL of your wit/LUIS/Dialogflow api calls to ``localhost:5000/parse``.

Rasa NLU is written in Python, but it you can use it from any language through :ref:`section_http`.
If your project *is* written in Python you can simply import the relevant classes.

Rasa is a set of tools for building more advanced bots, developed by `Rasa
<https://rasa.com>`_. This is the natural language understanding module. To
build conversational chatbots, you can interface Rasa NLU with libraries
that steer the flow of the conversation - more on this in :ref:`section_context`.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   tutorial

.. toctree::
   :maxdepth: 1
   :caption: User Documentation

   config
   migrating
   dataformat
   http
   python
   entities
   closeloop
   persist
   languages
   pipeline
   evaluation
   context
   faq
   migrations
   license

.. toctree::
   :maxdepth: 1
   :caption: Resources

   community

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   contribute
   changelog
