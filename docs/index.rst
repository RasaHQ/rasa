:desc: Learn more about open-source natural language processing library Rasa NLU
       for intent classification and entity extraction in on premise chatbots.

.. _index:

Build contextual chatbots and AI assistants with Rasa
=====================================================

.. note::
    These docs are for Rasa 1.0 and later. Docs for older versions are at http://legacy-docs.rasa.com.

    This is the documentation for version |release| of Rasa. Please make sure you are reading the documentation
    that matches the version you have installed.


Rasa is an open source machine learning framework for automated text and voice-based conversations.
Understand messages, hold conversations, and connect to messaging channels and APIs.


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   user-guide/installation
   user-guide/rasa-tutorial
   user-guide/command-line-interface
   user-guide/architecture
   user-guide/messaging-and-voice-channels
   user-guide/evaluating-models
   user-guide/running-the-server
   user-guide/running-rasa-with-docker
   user-guide/cloud-storage

.. toctree::
   :maxdepth: 1
   :caption: NLU
   :hidden:

   About <nlu/about>
   nlu/using-nlu-only
   nlu/training-data-format
   nlu/choosing-a-pipeline
   nlu/language-support
   nlu/entity-extraction
   nlu/components

.. toctree::
   :maxdepth: 1
   :caption: Core
   :hidden:

   About <core/about>
   core/stories
   core/domains
   core/responses
   core/actions
   core/policies
   core/slots
   core/forms
   core/interactive-learning
   core/fallback-actions

.. toctree::
   :maxdepth: 1
   :caption: Conversation Design
   :hidden:

   dialogue-elements/dialogue-elements
   dialogue-elements/small-talk
   dialogue-elements/completing-tasks
   dialogue-elements/guiding-users

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api/action-server
   api/http-api
   api/jupyter-notebooks
   api/agent
   api/custom-nlu-components
   api/events
   api/tracker
   api/tracker-stores
   api/event-brokers
   api/featurization
   migration-guide
   changelog

.. toctree:
   :maxdepth: 1
   :hidden:
   :caption: Migration (beta)

   migrate-from/google-dialogflow-to-rasa
