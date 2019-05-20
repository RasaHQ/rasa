:desc: Learn more about open-source natural language processing library Rasa NLU
       for intent classification and entity extraction in on premise chatbots.

.. _section_index:

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

   installation
   tutorial
   command-line-interface
   architecture
   channels
   evaluating-models
   server
   running-rasa-with-docker
   cloud-storage

.. toctree::
   :maxdepth: 1
   :caption: NLU
   :hidden:

   About <nlu/about>
   nlu/using-only-nlu
   nlu/data-format
   nlu/choosing-pipeline
   nlu/languages
   nlu/entities
   nlu/components

.. toctree::
   :maxdepth: 1
   :caption: Core
   :hidden:

   About <core/about>
   core/stories
   core/domains
   core/responses
   core/run-code-in-custom-actions
   core/policies
   core/slots
   core/slot-filling
   core/interactive-learning
   core/fallbacks


.. toctree::
   :maxdepth: 1
   :caption: Conversation Design
   :hidden:

   dialogue-elements/about
   dialogue-elements/small-talk
   dialogue-elements/completing-tasks
   dialogue-elements/guiding-users



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Reference

   api/action-server
   api/rasa-http-api
   api/jupyter
   api/agent 
   api/custom-nlu-components
   api/events
   api/tracker
   api/tracker-stores
   api/brokers
   api/featurizer

.. toctree:
   :maxdepth: 1
   :hidden:
   :caption: Migration (beta)

   migrate-from/google-dialogflow-to-rasa