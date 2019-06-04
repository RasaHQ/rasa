:desc: Rasa Changelog

Rasa Change Log
===============

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 1.0.

[Unreleased 1.0.6.aX] - `master`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added
-----

Changed
-------

Removed
-------

Fixed
-----
- slack notifications from bots correctly render text
- fixed usage of ``--log-file`` argument for ``rasa run`` and ``rasa shell``


[1.0.6] - 2019-06-03
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed backwards incompatible utils changes

[1.0.5] - 2019-06-03
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed spacy being a required dependency (regression)

[1.0.4] - 2019-06-03
^^^^^^^^^^^^^^^^^^^^

Added
-----
- automatic creation of index on the ``sender_id`` column when using an SQL
  tracker store. If you have an existing data and you are running into performance
  issues, please make sure to add an index manually using
  ``CREATE INDEX event_idx_sender_id ON events (sender_id);``.

Changed
-------
- NLU evaluation in cross-validation mode now also provides intent/entity reports,
  confusion matrix, etc.

[1.0.3] - 2019-05-30
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- non-ascii characters render correctly in stories generated from interactive learning
- validate domain file before usage, e.g. print proper error messages if domain file
  is invalid instead of raising errors

[1.0.2] - 2019-05-29
^^^^^^^^^^^^^^^^^^^^

Added
-----
- added ``domain_warnings()`` method to ``Domain`` which returns a dict containing the
  diff between supplied {actions, intents, entities, slots} and what's contained in the
  domain

Fixed
-----
- fix lookup table files failed to load issues/3622
- buttons can now be properly selected during cmdline chat or when in interactive learning
- set slots correctly when events are added through the API
- mapping policy no longer ignores NLU threshold
- mapping policy priority is correctly persisted


[1.0.1] - 2019-05-21
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- updated installation command in docs for Rasa X

[1.0.0] - 2019-05-21
^^^^^^^^^^^^^^^^^^^^

Added
-----
- added arguments to set the file paths for interactive training
- added quick reply representation for command-line output
- added option to specify custom button type for Facebook buttons
- added tracker store persisting trackers into a SQL database
  (``SQLTrackerStore``)
- added rasa command line interface and API
- Rasa Stack HTTP training endpoint at ``POST /jobs``. This endpoint
  will train a combined Rasa Core and NLU model
- ``ReminderCancelled(action_name)`` event to cancel given action_name reminder
  for current user
- Rasa Stack HTTP intent evaluation endpoint at ``POST /intentEvaluation``.
  This endpoints performs an intent evaluation of a Rasa Stack model
- option to create template for new utterance action in ``interactive learning``
- you can now choose actions previously created in the same session
  in ``interactive learning``
- add formatter 'black'
- channel-specific utterances via the ``- "channel":`` key in utterance templates
- arbitrary json messages via the ``- "custom":`` key in utterance templates and
  via ``utter_custom_json()`` method in custom actions
- support to load sub skills (domain, stories, nlu data)
- support to select which sub skills to load through ``import`` section in
  ``config.yml``
- support for spaCy 2.1
- a model for an agent can now also be loaded from a remote storage
- log level can be set via environment variable ``LOG_LEVEL``
- add ``--store-uncompressed`` to train command to not compress Rasa model
- log level of libraries, such as tensorflow, can be set via environment variable ``LOG_LEVEL_LIBRARIES``
- if no spaCy model is linked upon building a spaCy pipeline, an appropriate error message
  is now raised with instructions for linking one

Changed
-------
- renamed all CLI parameters containing any ``_`` to use dashes ``-`` instead (GNU standard)
- renamed ``rasa_core`` package to ``rasa.core``
- for interactive learning only include manually annotated and ner_crf entities in nlu export
- made ``message_id`` an additional argument to ``interpreter.parse``
- changed removing punctuation logic in ``WhitespaceTokenizer``
- ``training_processes`` in the Rasa NLU data router have been renamed to ``worker_processes``
- created a common utils package ``rasa.utils`` for nlu and core, common methods like ``read_yaml`` moved there
- removed ``--num_threads`` from run command (server will be asynchronous but
  running in a single thread)
- the ``_check_token()`` method in ``RasaChat`` now authenticates against ``/auth/verify`` instead of ``/user``
- removed ``--pre_load`` from run command (Rasa NLU server will just have a maximum of one model and that model will be
  loaded by default)
- changed file format of a stored trained model from the Rasa NLU server to ``tar.gz``
- train command uses fallback config if an invalid config is given
- test command now compares multiple models if a list of model files is provided for the argument ``--model``
- Merged rasa.core and rasa.nlu server into a single server. See swagger file in ``docs/_static/spec/server.yaml`` for
  available endpoints.
- ``utter_custom_message()`` method in rasa_core_sdk has been renamed to ``utter_elements()``
- updated dependencies. as part of this, models for spacy need to be reinstalled
  for 2.1 (from 2.0)
- make sure all command line arguments for ``rasa test`` and ``rasa interactive`` are actually used, removed arguments
  that were not used at all (e.g. ``--core`` for ``rasa test``)

Removed
-------
- removed possibility to execute ``python -m rasa_core.train`` etc. (e.g. scripts in ``rasa.core`` and ``rasa.nlu``).
  Use the CLI for rasa instead, e.g. ``rasa train core``.
- removed ``_sklearn_numpy_warning_fix`` from the ``SklearnIntentClassifier``
- removed ``Dispatcher`` class from core
- removed projects: the Rasa NLU server now has a maximum of one model at a time loaded.

Fixed
-----
- evaluating core stories with two stage fallback gave an error, trying to handle None for a policy
- the ``/evaluate`` route for the Rasa NLU server now runs evaluation
  in a parallel process, which prevents the currently loaded model unloading
- added missing implementation of the ``keys()`` function for the Redis Tracker
  Store
- in interactive learning: only updates entity values if user changes annotation
- log options from the command line interface are applied (they overwrite the environment variable)
- all message arguments (kwargs in dispatcher.utter methods, as well as template args) are now sent through to output channels
- utterance templates defined in actions are checked for existence upon training a new agent, and a warning
  is thrown before training if one is missing

.. _`master`: https://github.com/RasaHQ/rasa/

.. _`Semantic Versioning`: http://semver.org/
