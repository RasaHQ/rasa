
The Rasa Core dialogue engine
=============================

.. note::
   This is the documentation for version |release| of Rasa Core. Make sure you select
   the appropriate version of the documentation for your local installation!


Welcome to the Rasa Documentation!
----------------------------------

.. chat-bubble::
   :text: What am I looking at?
   :sender: user


.. chat-bubble::
   :text: Rasa is a framework for building conversational software:
      Messenger/Slack bots, Alexa skills, etc. We'll abbreviate this as a **bot**
      in this documentation. 
      Go `here <https://colab.research.google.com/github/RasaHQ/rasa_core/blob/master/getting_started.ipynb>`_ 
      to try it out without having to install anything.
   :sender: bot




*What's cool about it?*

    Rather than a bunch of ``if/else`` statements, the logic of your bot is
    based on a probabilistic model trained on example conversations.

*That sounds harder than writing a few if statements*

    In the beginning of a project, it's indeed easier to just hard code some logic.
    Rasa helps you when you want to go past that and create a bot that can handle more complexity.
    This `blog post <https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a>`_ explains the philosophy behind Rasa Core.

*Can I see it in action?*

    We thought you'd never ask!
    You can build a full example without installing anything on `colab <https://colab.research.google.com/github/RasaHQ/rasa_core/blob/master/getting_started.ipynb>`_.
    If you want to run stuff on your machine, follow the :ref:`installation`
    and check out :ref:`tutorial_basics` afterwards!


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installation
   tutorial_basics


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   connectors
   customactions
   slots
   debugging
   formfilling
   fallbacks
   evaluation
   faq

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   domains
   stories
   slots
   server
   http
   api/agent
   api/events
   api/tracker
   api/interpreter
   api/policy
   api/featurizer

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developer Documentation

   migrations
   changelog
