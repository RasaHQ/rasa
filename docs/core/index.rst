:desc: Get started with machine learning dialogue management to scale your bot
       development using Rasa Stack as a conversational AI platform.

.. _index:

The Rasa Core dialogue engine
=============================

.. note::
   This is the documentation for version |release| of Rasa Core. Make sure you select
   the appropriate version of the documentation for your local installation!


.. chat-bubble::
   :text: What am I looking at?
   :sender: user


.. chat-bubble::
   :text: Rasa is a framework for building conversational software:
      Messenger/Slack bots, Alexa skills, etc. We'll abbreviate this as a <strong>bot</strong>
      in this documentation.
   :sender: bot

.. chat-bubble::
   :text: What's cool about it?
   :sender: user

.. chat-bubble::
   :text: Rather than a bunch of <code>if/else</code> statements, the logic of your bot is
      based on a machine learning model trained on example conversations.
   :sender: bot

.. chat-bubble::
   :text: That sounds harder than writing a few if statements.
   :sender: user


.. chat-bubble::
   :text: In the beginning of a project, it seems easier to just hard code some logic.
      Rasa helps you when you want to go past that and create a bot that can handle more complexity.
      This <a href="https://medium.com/rasa-blog/a-new-approach-to-conversational-software-2e64a5d05f2a" target="_blank">blog post </a> explains the philosophy behind Rasa Core.
   :sender: bot


.. chat-bubble::
   :text: Can I see it in action?
   :sender: user

.. chat-bubble::
   :text: We thought you'd never ask!
      You can build a full example without installing anything, just go to the quickstart!
   :sender: bot


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   quickstart
   installation

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   architecture
   connectors
   run-code-in-custom-actions
   slots
   slotfilling
   responses
   interactive_learning
   fallbacks
   policies
   debugging
   evaluation
   docker_walkthrough

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   domains
   stories
   api/slots_api
   server
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
   tracker_stores
   brokers
   docker
   old_core_changelog
   support
