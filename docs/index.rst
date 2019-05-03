:desc: Learn more about open-source natural language processing library Rasa NLU
       for intent classification and entity extraction in on premise chatbots.

.. _section_index:

Build contextual chatbots and AI assistants with Rasa
=====================================================

.. note::
    This is the documentation for version |release| of Rasa. Please make sure you are reading the documentation
    that matches the version you have installed.


Rasa is an open source machine learning framework for automated text and voice-based conversations.
Understand messages, hold conversations, and connect to messaging channels and APIs.


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   project-structure
   command-line-interface
   architecture
   messaging-and-voice-channels
   evaluating-models


.. toctree::
   :maxdepth: 1
   :caption: Dialogue Elements

   dialogue-elements/small-talk
   dialogue-elements/completing-tasks
   dialogue-elements/guiding-users


.. toctree::
   :maxdepth: 1
   :caption: NLU
   :hidden:

   nlu/data-format
   nlu/choosing-pipeline
   nlu/languages
   nlu/entities
   nlu/components

.. toctree::
   :maxdepth: 1
   :caption: Core
   :hidden:

   core/domains
   core/stories
   core/responses
   core/run-code-in-custom-actions
   core/policies
   core/slots
   core/interactive-learning
   core/slot-filling
   core/fallbacks
   core/evaluation



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