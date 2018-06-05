
Language Understanding with Rasa NLU
====================================

.. note::
    This is the documentation for version |release| of Rasa NLU. Please make sure you are reading the documentation 
    that matches the version you have installed.


.. button:: 
   :text: Button Text
   :link: link/to/stuff.html

.. card:: 
   :title: Turn Natural Language into Structured Data
   :subtitle: NLU Quickstart
   :link: link/to/stuff.html
   :description: Convert messages into intents and entities and live the actual dream.
   :image_url: http://core.rasa.com/_images/mood_bot.png


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


The intended audience is mainly people developing chatbots and voice apps.

Who Uses Rasa?
~~~~~~~~~~~~~~

Rasa NLU and Rasa Core together form the `Rasa Stack <https://rasa.com/products/rasa-stack>`_, which is used in production 
at many Fortune 500 companies and startups,
and is also used by R&D labs and research groups.

There is a big community of Rasa Developers, with hundreds of external contributors. 

How Does Rasa Compare to Alternatives?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use Rasa as a drop-in replacement for `wit <https://wit.ai>`_ , `LUIS <https://www.luis.ai>`_ , or `Dialogflow <https://dialogflow.com>`_, see here for details.

You can read about the advantages of using open source NLU in this `blog post <https://medium.com/rasa-blog/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d>`_ .
You can see independent benchmarks comparing Rasa NLU to various closed source tools, `here <link1>`_ and `here <link1>`_. 


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   quickstart


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   choosing_pipeline
   entities
   languages
   evaluation
   faq

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   dataformat
   pipeline
   config
   http
   python
   persist
   docker

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developer Documentation

   customcomponents
   migrations
   license
   changelog
