:desc: Rasa Changelog

Rasa Change Log
===============

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 1.0.

[Unreleased 1.0.0.aX] - `master`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
- add ``rasa interactive core`` to command line interface
- support for spaCy 2.1
- a model for an agent can now also be loaded from a remote storage
- log level can be set via environment variable ``LOG_LEVEL``

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
- ``rasa train`` uses fallback config if an invalid config is given
- ``rasa test core`` compares multiple models if a list of model files is provided for the argument ``--model``
- ``rasa train`` falls back to ``rasa train core``/``rasa train nlu`` if the corresponding training data are missing
- Merged rasa.core and rasa.nlu server into a single server. See swagger file in ``docs/_static/spec/server.yaml`` for
  available endpoints.
- ``utter_custom_message()`` method in rasa_core_sdk has been renamed to ``utter_elements()``


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
