:desc: Rasa Changelog


Rasa Change Log
===============

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 1.0.

[Unreleased 1.3.10]
^^^^^^^^^^^^^^^^^^^

Added
-----

Fixed
-----

Changed
-------

Removed
-------

[1.3.9] - 2019-10-10
^^^^^^^^^^^^^^^^^^^^

Added
-----
- Port of 1.2.10 (support for RabbitMQ TLS authentication and ``port`` key in
  event broker endpoint config).
- Port of 1.2.11 (support for passing a CA file for SSL certificate verification via the
  --ssl-ca-file flag).

Fixed
-----
- Fixed the hanging HTTP call with ``ner_duckling_http`` pipeline.
- Fixed text processing of ``intent`` attribute inside ``CountVectorFeaturizer``.

[1.3.8] - 2019-10-08
^^^^^^^^^^^^^^^^^^^^

Changed
-------
- Policies now only get imported if they are actually used. This removes
  TensorFlow warnings when starting Rasa X

Fixed
-----
- Fixed error ``Object of type 'MaxHistoryTrackerFeaturizer' is not JSON serializable``
  when running ``rasa train core``
- Default channel ``send_`` methods no longer support kwargs as they caused issues in incompatible channels
- Fixed ``argument of type 'NoneType' is not iterable`` when using ``rasa shell``,
  ``rasa interactive`` / ``rasa run``

[1.3.7] - 2019-09-27
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- re-added TLS, SRV dependencies for PyMongo
- socketio can now be run without turning on the ``--enable-api`` flag
- MappingPolicy no longer fails when the latest action doesn't have a policy

[1.3.6] - 2019-09-21
^^^^^^^^^^^^^^^^^^^^

Added
-----
- Added the ability for users to specify a conversation id to send a message to when
  using the ``RasaChat`` input channel.

[1.3.5] - 2019-09-20
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Fixed issue where ``rasa init`` would fail without spaCy being installed

[1.3.4] - 2019-09-20
^^^^^^^^^^^^^^^^^^^^

Added
-----
- Added the ability to set the ``backlog`` parameter in Sanics ``run()`` method using
  the ``SANIC_BACKLOG`` environment variable. This parameter sets the
  number of unaccepted connections the server allows before refusing new
  connections. A default value of 100 is used if the variable is not set.
- Status endpoint (``/status``) now also returns the number of training processes currently running

Fixed
-----
- Added the ability to properly deal with spaCy ``Doc``-objects created on
  empty strings as discussed `here <https://github.com/RasaHQ/rasa/issues/4445>`_.
  Only training samples that actually bear content are sent to ``self.nlp.pipe``
  for every given attribute. Non-content-bearing samples are converted to empty
  ``Doc``-objects. The resulting lists are merged with their preserved order and
  properly returned.
- asyncio warnings are now only printed if the callback takes more than 100ms
  (up from 1ms).
- ``agent.load_model_from_server`` no longer affects logging.

Changed
-------
- The endpoint ``POST /model/train`` no longer supports specifying an output directory
  for the trained model using the field ``out``. Instead you can choose whether you
  want to save the trained model in the default model directory (``models``)
  (default behavior) or in a temporary directory by specifying the
  ``save_to_default_model_directory`` field in the training request.

[1.3.3] - 2019-09-13
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Added a check to avoid training CountVectorizer for a particular attribute of a message if no text is provided for that attribute across the training data.
- Default one-hot representation for label featurization inside ``EmbeddingIntentClassifier`` if label features don't exist.
- Policy ensemble no longer incorrectly wrings "missing mapping policy" when
  mapping policy is present.
- "test" from ``utter_custom_json`` now correctly saved to tracker when using telegram channel

Removed
-------
- Removed computation of ``intent_spacy_doc``. As a result, none of the spacy components process intents now.

[1.3.2] - 2019-09-10
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- SQL tracker events are retrieved ordered by timestamps. This fixes interactive
  learning events being shown in the wrong order.

[1.3.1] - 2019-09-09
^^^^^^^^^^^^^^^^^^^^

Changed
-------
- Pin gast to == 0.2.2


[1.3.0] - 2019-09-05
^^^^^^^^^^^^^^^^^^^^

Added
-----
- Added option to persist nlu training data (default: False)
- option to save stories in e2e format for interactive learning
- bot messages contain the ``timestamp`` of the ``BotUttered`` event, which can be used in channels
- ``FallbackPolicy`` can now be configured to trigger when the difference between confidences of two predicted intents is too narrow
- experimental training data importer which supports training with data of multiple
  sub bots. Please see the
  `docs <https://rasa.com/docs/rasa/api/training-data-importers/>`_ for more
  information.
- throw error during training when triggers are defined in the domain without
  ``MappingPolicy`` being present in the policy ensemble
- The tracker is now available within the interpreter's ``parse`` method, giving the
  ability to create interpreter classes that use the tracker state (eg. slot values)
  during the parsing of the message. More details on motivation of this change see
  issues/3015.
- add example bot ``knowledgebasebot`` to showcase the usage of ``ActionQueryKnowledgeBase``
- ``softmax`` starspace loss for both ``EmbeddingPolicy`` and ``EmbeddingIntentClassifier``
- ``balanced`` batching strategy for both ``EmbeddingPolicy`` and ``EmbeddingIntentClassifier``
- ``max_history`` parameter for ``EmbeddingPolicy``
- Successful predictions of the NER are written to a file if ``--successes`` is set when running ``rasa test nlu``
- Incorrect predictions of the NER are written to a file by default. You can disable it via ``--no-errors``.
- New NLU component ``ResponseSelector`` added for the task of response selection
- Message data attribute can contain two more keys - ``response_key``, ``response`` depending on the training data
- New action type implemented by ``ActionRetrieveResponse`` class and identified with ``response_`` prefix
- Vocabulary sharing inside ``CountVectorsFeaturizer`` with ``use_shared_vocab`` flag. If set to True, vocabulary of corpus is shared between text, intent and response attributes of message
- Added an option to share the hidden layer weights of text input and label input inside ``EmbeddingIntentClassifier`` using the flag ``share_hidden_layers``
- New type of training data file in NLU which stores response phrases for response selection task.
- Add flag ``intent_split_symbol`` and ``intent_tokenization_flag`` to all ``WhitespaceTokenizer``, ``JiebaTokenizer`` and ``SpacyTokenizer``
- Added evaluation for response selector. Creates a report ``response_selection_report.json`` inside ``--out`` directory.
- argument ``--config-endpoint`` to specify the URL from which ``rasa x`` pulls
  the runtime configuration (endpoints and credentials)
- ``LockStore`` class storing instances of ``TicketLock`` for every ``conversation_id``
- environment variables ``SQL_POOL_SIZE`` (default: 50) and ``SQL_MAX_OVERFLOW``
  (default: 100) can be set to control the pool size and maximum pool overflow for
  ``SQLTrackerStore`` when used with the ``postgresql`` dialect
- Add a `bot_challenge` intent and a `utter_iamabot` action to all example projects and the rasa init bot.
- Allow sending attachments when using the socketio channel
- ``rasa data validate`` will fail with a non-zero exit code if validation fails

Changed
-------
- added character-level ``CountVectorsFeaturizer`` with empirically found parameters
  into the ``supervised_embeddings`` NLU pipeline template
- NLU evaluations now also stores its output in the output directory like the core evaluation
- show warning in case a default path is used instead of a provided, invalid path
- compare mode of ``rasa train core`` allows the whole core config comparison,
  naming style of models trained for comparison is changed (this is a breaking change)
- pika keeps a single connection open, instead of open and closing on each incoming event
- ``RasaChatInput`` fetches the public key from the Rasa X API. The key is used to
  decode the bearer token containing the conversation ID. This requires
  ``rasa-x>=0.20.2``.
- more specific exception message when loading custom components depending on whether component's path or
  class name is invalid or can't be found in the global namespace
- change priorities so that the ``MemoizationPolicy`` has higher priority than the ``MappingPolicy``
- substitute LSTM with Transformer in ``EmbeddingPolicy``
- ``EmbeddingPolicy`` can now use ``MaxHistoryTrackerFeaturizer``
- non zero ``evaluate_on_num_examples`` in ``EmbeddingPolicy``
  and ``EmbeddingIntentClassifier`` is the size of
  hold out validation set that is excluded from training data
- defaults parameters and architectures for both ``EmbeddingPolicy`` and
  ``EmbeddingIntentClassifier`` are changed (this is a breaking change)
- evaluation of NER does not include 'no-entity' anymore
- ``--successes`` for ``rasa test nlu`` is now boolean values. If set incorrect/successful predictions
  are saved in a file.
- ``--errors`` is renamed to ``--no-errors`` and is now a boolean value. By default incorrect predictions are saved
  in a file. If ``--no-errors`` is set predictions are not written to a file.
- Remove ``label_tokenization_flag`` and ``label_split_symbol`` from ``EmbeddingIntentClassifier``. Instead move these parameters to ``Tokenizers``.
- Process features of all attributes of a message, i.e. - text, intent and response inside the respective component itself. For e.g. - intent of a message is now tokenized inside the tokenizer itself.
- Deprecate ``as_markdown`` and ``as_json`` in favour of ``nlu_as_markdown`` and ``nlu_as_json`` respectively.
- pin python-engineio >= 3.9.3
- update python-socketio req to >= 4.3.1

Fixed
-----
- ``rasa test nlu`` with a folder of configuration files
- ``MappingPolicy`` standard featurizer is set to ``None``
- Removed ``text`` parameter from send_attachment function in slack.py to avoid duplication of text output to slackbot
- server ``/status`` endpoint reports status when an NLU-only model is loaded

Removed
-------
- Removed ``--report`` argument from ``rasa test nlu``. All output files are stored in the ``--out`` directory.

[1.2.11] - 2019-10-09
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- Support for passing a CA file for SSL certificate verification via the
  --ssl-ca-file flag

[1.2.10] - 2019-10-08
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- Added support for RabbitMQ TLS authentication. The following environment variables
  need to be set:
  ``RABBITMQ_SSL_CLIENT_CERTIFICATE`` - path to the SSL client certificate (required)
  ``RABBITMQ_SSL_CLIENT_KEY`` - path to the SSL client key (required)
  ``RABBITMQ_SSL_CA_FILE`` - path to the SSL CA file (optional, for certificate
  verification)
  ``RABBITMQ_SSL_KEY_PASSWORD`` - SSL private key password (optional)
- Added ability to define the RabbitMQ port using the ``port`` key in the
  ``event_broker`` endpoint config.

[1.2.9] - 2019-09-17
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Correctly pass SSL flag values to x CLI command (backport of


[1.2.8] - 2019-09-10
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- SQL tracker events are retrieved ordered by timestamps. This fixes interactive
  learning events being shown in the wrong order. Backport of ``1.3.2`` patch
  (PR #4427).


[1.2.7] - 2019-09-02
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Added ``query`` dictionary argument to ``SQLTrackerStore`` which will be appended
  to the SQL connection URL as query parameters.


[1.2.6] - 2019-09-02
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed bug that occurred when sending template ``elements`` through a channel that doesn't support them

[1.2.5] - 2019-08-26
^^^^^^^^^^^^^^^^^^^^

Added
-----
- SSL support for ``rasa run`` command. Certificate can be specified using
  ``--ssl-certificate`` and ``--ssl-keyfile``.

Fixed
-----
- made default augmentation value consistent across repo
- ``'/restart'`` will now also restart the bot if the tracker is paused


[1.2.4] - 2019-08-23
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- the ``SocketIO`` input channel now allows accesses from other origins
  (fixes ``SocketIO`` channel on Rasa X)

[1.2.3] - 2019-08-15
^^^^^^^^^^^^^^^^^^^^

Changed
-------
- messages with multiple entities are now handled properly with e2e evaluation
- ``data/test_evaluations/end_to_end_story.md`` was re-written in the
  restaurantbot domain

[1.2.3] - 2019-08-15
^^^^^^^^^^^^^^^^^^^^

Changed
-------
- messages with multiple entities are now handled properly with e2e evaluation
- ``data/test_evaluations/end_to_end_story.md`` was re-written in the restaurantbot domain

Fixed
-----
- Free text input was not allowed in the Rasa shell when the response template
  contained buttons, which has now been fixed.

[1.2.2] - 2019-08-07
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- ``UserUttered`` events always got the same timestamp

[1.2.1] - 2019-08-06
^^^^^^^^^^^^^^^^^^^^

Added
-----
- Docs now have an ``EDIT THIS PAGE`` button

Fixed
-----
- ``Flood control exceeded`` error in Telegram connector which happened because the
  webhook was set twice

[1.2.0] - 2019-08-01
^^^^^^^^^^^^^^^^^^^^

Added
-----
- add root route to server started without ``--enable-api`` parameter
- add ``--evaluate-model-directory`` to ``rasa test core`` to evaluate models
  from ``rasa train core -c <config-1> <config-2>``
- option to send messages to the user by calling
  ``POST /conversations/{conversation_id}/execute``

Changed
-------
- ``Agent.update_model()`` and ``Agent.handle_message()`` now work without needing to set a domain
  or a policy ensemble
- Update pytype to ``2019.7.11``
- new event broker class: ``SQLProducer``. This event broker is now used when running locally with
  Rasa X
- API requests are not longer logged to ``rasa_core.log`` by default in order to avoid
  problems when running on OpenShift (use ``--log-file rasa_core.log`` to retain the
  old behavior)
- ``metadata`` attribute added to ``UserMessage``

Fixed
-----
- ``rasa test core`` can handle compressed model files
- rasa can handle story files containing multi line comments
- template will retain `{` if escaped with `{`. e.g. `{{"foo": {bar}}}` will result in `{"foo": "replaced value"}`

[1.1.8] - 2019-07-25
^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``TrainingFileImporter`` interface to support customizing the process of loading
  training data
- fill slots for custom templates

Changed
-------
- ``Agent.update_model()`` and ``Agent.handle_message()`` now work without needing to set a domain
  or a policy ensemble
- update pytype to ``2019.7.11``

Fixed
-----
- interactive learning bug where reverted user utterances were dumped to training data
- added timeout to terminal input channel to avoid freezing input in case of server
  errors
- fill slots for image, buttons, quick_replies and attachments in templates
- ``rasa train core`` in comparison mode stores the model files compressed (``tar.gz`` files)
- slot setting in interactive learning with the TwoStageFallbackPolicy


[1.1.7] - 2019-07-18
^^^^^^^^^^^^^^^^^^^^

Added
-----
- added optional pymongo dependencies ``[tls, srv]`` to ``requirements.txt`` for better mongodb support
- ``case_sensitive`` option added to ``WhiteSpaceTokenizer`` with ``true`` as default.

Fixed
-----
- validation no longer throws an error during interactive learning
- fixed wrong cleaning of ``use_entities`` in case it was a list and not ``True``
- updated the server endpoint ``/model/parse`` to handle also messages with the intent prefix
- fixed bug where "No model found" message appeared after successfully running the bot
- debug logs now print to ``rasa_core.log`` when running ``rasa x -vv`` or ``rasa run -vv``

[1.1.6] - 2019-07-12
^^^^^^^^^^^^^^^^^^^^

Added
-----
- rest channel supports setting a message's input_channel through a field
  ``input_channel`` in the request body

Changed
-------
- recommended syntax for empty ``use_entities`` and ``ignore_entities`` in the domain file
  has been updated from ``False`` or ``None`` to an empty list (``[]``)

Fixed
-----
- ``rasa run`` without ``--enable-api`` does not require a local model anymore
- using ``rasa run`` with ``--enable-api`` to run a server now prints
  "running Rasa server" instead of "running Rasa Core server"
- actions, intents, and utterances created in ``rasa interactive`` can no longer be empty


[1.1.5] - 2019-07-10
^^^^^^^^^^^^^^^^^^^^

Added
-----
- debug logging now tells you which tracker store is connected
- the response of ``/model/train`` now includes a response header for the trained model filename
- ``Validator`` class to help developing by checking if the files have any errors
- project's code is now linted using flake8
- ``info`` log when credentials were provided for multiple channels and channel in
  ``--connector`` argument was specified at the same time
- validate export paths in interactive learning

Changed
-------
- deprecate ``rasa.core.agent.handle_channels(...)`. Please use ``rasa.run(...)``
  or ``rasa.core.run.configure_app`` instead.
- ``Agent.load()`` also accepts ``tar.gz`` model file

Removed
-------
- revert the stripping of trailing slashes in endpoint URLs since this can lead to
  problems in case the trailing slash is actually wanted
- starter packs were removed from Github and are therefore no longer tested by Travis script

Fixed
-----
- all temporal model files are now deleted after stopping the Rasa server
- ``rasa shell nlu`` now outputs unicode characters instead of ``\uxxxx`` codes
- fixed PUT /model with model_server by deserializing the model_server to
  EndpointConfig.
- ``x in AnySlotDict`` is now ``True`` for any ``x``, which fixes empty slot warnings in
  interactive learning
- ``rasa train`` now also includes NLU files in other formats than the Rasa format
- ``rasa train core`` no longer crashes without a ``--domain`` arg
- ``rasa interactive`` now looks for endpoints in ``endpoints.yml`` if no ``--endpoints`` arg is passed
- custom files, e.g. custom components and channels, load correctly when using
  the command line interface
- ``MappingPolicy`` now works correctly when used as part of a PolicyEnsemble


[1.1.4] - 2019-06-18
^^^^^^^^^^^^^^^^^^^^

Added
-----
- unfeaturize single entities
- added agent readiness check to the ``/status`` resource

Changed
-------
- removed leading underscore from name of '_create_initial_project' function.

Fixed
-----
- fixed bug where facebook quick replies were not rendering
- take FB quick reply payload rather than text as input
- fixed bug where `training_data` path in `metadata.json` was an absolute path

[1.1.3] - 2019-06-14
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed any inconsistent type annotations in code and some bugs revealed by
  type checker

[1.1.2] - 2019-06-13
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed duplicate events appearing in tracker when using a PostgreSQL tracker store

[1.1.1] - 2019-06-13
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed compatibility with Rasa SDK
- bot responses can contain ``custom`` messages besides other message types

[1.1.0] - 2019-06-13
^^^^^^^^^^^^^^^^^^^^

Added
-----
- nlu configs can now be directly compared for performance on a dataset
  in ``rasa test nlu``

Changed
-------
- update the tracker in interactive learning through reverting and appending events
  instead of replacing the tracker
- ``POST /conversations/{conversation_id}/tracker/events`` supports a list of events

Fixed
-----
- fixed creation of ``RasaNLUHttpInterpreter``
- form actions are included in domain warnings
- default actions, which are overriden by custom actions and are listed in the
  domain are excluded from domain warnings
- SQL ``data`` column type to ``Text`` for compatibility with MySQL
- non-featurizer training parameters don't break `SklearnPolicy` anymore

[1.0.9] - 2019-06-10
^^^^^^^^^^^^^^^^^^^^

Changed
-------
- revert PR #3739 (as this is a breaking change): set ``PikaProducer`` and
  ``KafkaProducer`` default queues back to ``rasa_core_events``

[1.0.8] - 2019-06-10
^^^^^^^^^^^^^^^^^^^^

Added
-----
- support for specifying full database urls in the ``SQLTrackerStore`` configuration
- maximum number of predictions can be set via the environment variable
  ``MAX_NUMBER_OF_PREDICTIONS`` (default is 10)

Changed
-------
- default ``PikaProducer`` and ``KafkaProducer`` queues to ``rasa_production_events``
- exclude unfeaturized slots from domain warnings

Fixed
-----
- loading of additional training data with the ``SkillSelector``
- strip trailing slashes in endpoint URLs

[1.0.7] - 2019-06-06
^^^^^^^^^^^^^^^^^^^^

Added
-----
- added argument ``--rasa-x-port`` to specify the port of Rasa X when running Rasa X locally via ``rasa x``

Fixed
-----
- slack notifications from bots correctly render text
- fixed usage of ``--log-file`` argument for ``rasa run`` and ``rasa shell``
- check if correct tracker store is configured in local mode

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
- Rasa  HTTP training endpoint at ``POST /jobs``. This endpoint
  will train a combined Rasa Core and NLU model
- ``ReminderCancelled(action_name)`` event to cancel given action_name reminder
  for current user
- Rasa HTTP intent evaluation endpoint at ``POST /intentEvaluation``.
  This endpoints performs an intent evaluation of a Rasa model
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
