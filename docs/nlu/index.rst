:desc: Learn more about open-source natural language processing library Rasa NLU
       for intent classification and entity extraction in on premise chatbots.

.. _section_index:

Rasa NLU: Language Understanding for chatbots and AI assistants
===============================================================

.. note::
    This is the documentation for version |release| of Rasa NLU. Please make sure you are reading the documentation
    that matches the version you have installed.



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


The target audience is developers building chatbots and voice apps.

The main reasons for using open source NLU are that:

- you don't have to hand over all your training data to Google, Microsoft, Amazon, or Facebook.
- Machine Learning is not one-size-fits all. You can tweak and customize models for your training data.
- Rasa NLU runs wherever you want, so you don't have to make an extra network request for every message that comes in.

You can read about the advantages of using open source NLU in this `blog post <https://medium.com/rasa-blog/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d>`_ .
You can see an independent benchmark comparing Rasa NLU to closed source alternatives `here <https://drive.google.com/file/d/0B0l-QQUtZzsdVEpaWEpyVzhZQzQ/view>`_.

.. include:: feedback.inc

.. toctree::
   :maxdepth: 1
   :caption: Getting Started
   :hidden:

   Try It Out <quickstart>
   installation


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   choosing_pipeline
   languages
   entities
   evaluation
   fallback
   faq

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   dataformat
   components
   config
   http
   python
   persist
   endpoint_configuration
   docker

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Documentation

   customcomponents
   migrations
   license
   old_nlu_changelog
   support
