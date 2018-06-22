
Language Understanding with Rasa NLU
====================================

.. note::
    This is the documentation for version |release| of Rasa NLU. Please make sure you are reading the documentation 
    that matches the version you have installed.



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
You can read about the advantages of using open source NLU in this `blog post <https://medium.com/rasa-blog/do-it-yourself-nlp-for-bot-developers-2e2da2817f3d>`_ .
You can see an independent benchmark comparing Rasa NLU to various closed source tools `here <https://drive.google.com/file/d/0B0l-QQUtZzsdVEpaWEpyVzhZQzQ/view>`_. 


.. toctree::
   :maxdepth: 1
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
   fallback
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
   :maxdepth: 2
   :hidden:
   :caption: Developer Documentation

   customcomponents
   migrations
   license
   changelog
