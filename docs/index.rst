
The Rasa dialogue engine
========================

.. note::
   This is the documentation for version |release| of Rasa Core. Make sure you select
   the appropriate version of the documentation for your local installation!


Welcome to the Rasa Documentation!
----------------------------------

*what am I looking at?*

    Rasa is a python framework for building conversational software: Messenger/Slack bots, Alexa skills, etc. We'll abbreviate this as a 'bot' in this documentation.

*what's cool about it?*

    Rather than a bunch of ``if/else`` statements, the logic of your bot is based on a probabilistic model trained on example conversations.

*that sounds harder than writing a few if statements*

    In the beginning of a project, it's indeed easier to just hard code some logic. 
    Rasa helps you when you want to go past that and create a bot that can handle more complexity.

*Can I see it in action?*

    We thought you'd never ask! Check out :ref:`tutorial_scratch` .



.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   read_first
   installation
   tour
   tutorial_scratch

.. toctree::
   :maxdepth: 1
   :caption: More Tutorials

   tutorial_babi
   tutorial_fake_user

.. toctree::
   :maxdepth: 1
   :caption: Deep Dives

   patterns
   http
   slot_types
   plumbing
   connectors
   domains_actions
   state
   policies
   stories
   scheduling
   featurisation
   interpreters
   message_handling

.. toctree::
   :maxdepth: 1
   :caption: API

   api/agent
   api/events

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   changelog
