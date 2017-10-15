
The Rasa dialogue engine
========================

.. note::
   This is the documentation for version |release| of Rasa Core. Make sure you select
   the appropriate version of the documentation for your local installation!


Welcome to the Rasa Documentation!
----------------------------------

*what am I looking at?*

    Rasa is a framework for building conversational software:
    Messenger/Slack bots, Alexa skills, etc. We'll abbreviate this as a 'bot'
    in this documentation. You can

    - write your bots logic in python code (recommended),
    - or use Rasa Core as a webservice (experimental, see :ref:`section_http`).

*what's cool about it?*

    Rather than a bunch of ``if/else`` statements, the logic of your bot is
    based on a probabilistic model trained on example conversations.

*that sounds harder than writing a few if statements*

    In the beginning of a project, it's indeed easier to just hard code some logic. 
    Rasa helps you when you want to go past that and create a bot that can handle more complexity.

*Can I see it in action?*

    We thought you'd never ask! Check out :ref:`tutorial_scratch` .



.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   read_first
   no_python
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
   plumbing
   http
   domains
   connectors
   stories
   scheduling

.. toctree::
   :maxdepth: 1
   :caption: Python API

   api/agent
   api/events

.. toctree::
   :maxdepth: 1
   :caption: Developer Documentation

   featurisation
   interpreters
   policies
   state
   changelog
