
The Rasa dialogue engine
========================

.. note::
   This is the documentation for version |release| of Rasa Core. Make sure you select
   the appropriate version of the documentation for your local installation!


Welcome to the Rasa Documentation!
----------------------------------

*What am I looking at?*

    Rasa is a framework for building conversational software:
    Messenger/Slack bots, Alexa skills, etc. We'll abbreviate this as a **bot**
    in this documentation. You can

    - implement the actions your bot can take in python code (recommended),
    - or use Rasa Core as a webservice (experimental, see :ref:`section_http`).

*What's cool about it?*

    Rather than a bunch of ``if/else`` statements, the logic of your bot is
    based on a probabilistic model trained on example conversations.

*That sounds harder than writing a few if statements*

    In the beginning of a project, it's indeed easier to just hard code some logic.
    Rasa helps you when you want to go past that and create a bot that can handle more complexity.

*Can I see it in action?*

    We thought you'd never ask! Make sure to follow the :ref:`installation`
    and check out :ref:`tutorial_basics` afterwards!


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   motivation
   no_python
   installation

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorial_basics
   tutorial_supervised
   tutorial_interactive_learning
   tutorial_remote

.. toctree::
   :maxdepth: 1
   :caption: Deep Dives

   domains
   stories
   patterns
   plumbing
   http
   connectors
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
   migrations
   changelog
