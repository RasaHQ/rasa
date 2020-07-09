:desc: Rasa Open Source Changelog


Rasa Open Source Change Log
===========================

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 1.0.

..
    You should **NOT** be adding new change log entries to this file, this
    file is managed by ``towncrier``.

    You **may** edit previous change logs to fix problems like typo corrections or such.
    You can find more information on how to add a new change log entry at
    https://github.com/RasaHQ/rasa/tree/master/changelog/ .

.. towncrier release notes start

[1.10.5] - 2020-07-02
^^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#6119 <https://github.com/rasahq/rasa/issues/6119>`_: Explicitly remove all emojis which appear as unicode characters from the output of ``regex.sub`` inside ``WhitespaceTokenizer``.


[1.10.4] - 2020-07-01
^^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5998 <https://github.com/rasahq/rasa/issues/5998>`_: ``WhitespaceTokenizer`` does not remove vowel signs in Hindi anymore.
- `#6031 <https://github.com/rasahq/rasa/issues/6031>`_: Previously, specifying a lock store in the endpoint configuration with a type other than ``redis`` or ``in_memory``
  would lead to an ``AttributeError: 'str' object has no attribute 'type'``. This bug is fixed now.
- `#6032 <https://github.com/rasahq/rasa/issues/6032>`_: Fix ``Interpreter parsed an intent ...`` warning when using the ``/model/parse`` 
  endpoint with an NLU-only model.
- `#6042 <https://github.com/rasahq/rasa/issues/6042>`_: Convert entity values coming from any entity extractor to string during evaluation to avoid mismatches due to
  different types.
- `#6078 <https://github.com/rasahq/rasa/issues/6078>`_: The assistant will respond through the `webex` channel to any user (room) communicating to it. Before the bot responded only to a fixed ``roomId`` set in the ``credentials.yml`` config file.


[1.10.3] - 2020-06-12
^^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- `#3900 <https://github.com/rasahq/rasa/issues/3900>`_: Reduced duplicate logs and warnings when running ``rasa train``.

Bugfixes
--------
- `#5972 <https://github.com/rasahq/rasa/issues/5972>`_: Remove the ``clean_up_entities`` method from the ``DIETClassifier`` and ``CRFEntityExtractor`` as it let to incorrect
  entity predictions.
- `#5976 <https://github.com/rasahq/rasa/issues/5976>`_: Fix server crashes that occurred when Rasa Open Source pulls a model from a
  :ref:`model server <server_fetch_from_server>` and an exception was thrown during
  model loading (such as a domain with invalid YAML).


[1.10.2] - 2020-06-03
^^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5521 <https://github.com/rasahq/rasa/issues/5521>`_: Responses used in ResponseSelector now support new lines with explicitly adding ``\n`` between them.
- `#5758 <https://github.com/rasahq/rasa/issues/5758>`_: Fixed a bug in `rasa export <https://rasa.com/docs/rasa-x/installation-and-setup/deploy#connect-rasa-deployment>`_ (:ref:`section_export`) which caused Rasa Open Source to only migrate conversation events from the last :ref:`session_config`.


[1.10.1] - 2020-05-15
^^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- `#5794 <https://github.com/rasahq/rasa/issues/5794>`_: Creating a ``Domain`` using ``Domain.fromDict`` can no longer alter the input dictionary.
  Previously, there could be problems when the input dictionary was re-used for other
  things after creating the ``Domain`` from it.

Bugfixes
--------
- `#5617 <https://github.com/rasahq/rasa/issues/5617>`_: Don't create TensorBoard log files during prediction.
- `#5638 <https://github.com/rasahq/rasa/issues/5638>`_: Fix: DIET breaks with empty spaCy model
- `#5755 <https://github.com/rasahq/rasa/issues/5755>`_: Remove ``clean_up_entities`` from extractors that extract pre-defined entities.
  Just keep the clean up method for entity extractors that extract custom entities.
- `#5792 <https://github.com/rasahq/rasa/issues/5792>`_: Fixed issue where the ``DucklingHTTPExtractor`` component would
  not work if its `url` contained a trailing slash.
- `#5825 <https://github.com/rasahq/rasa/issues/5825>`_: Fix list index out of range error in ``ensure_consistent_bilou_tagging``.

Miscellaneous internal changes
------------------------------
- #5788


[1.10.0] - 2020-04-28
^^^^^^^^^^^^^^^^^^^^^

Features
--------
- `#3765 <https://github.com/rasahq/rasa/issues/3765>`_: Add support for entities with roles and grouping of entities in Rasa NLU.

  You can now define a role and/or group label in addition to the entity type for entities.
  Use the role label if an entity can play different roles in your assistant.
  For example, a city can be a destination or a departure city.
  The group label can be used to group multiple entities together.
  For example, you could group different pizza orders, so that you know what toppings goes with which pizza and
  what size which pizza has.
  For more details see :ref:`entities-roles-groups`.

  To fill slots from entities with a specific role/group, you need to either use forms or use a custom action.
  We updated the tracker method ``get_latest_entity_values`` to take an optional role/group label.
  If you want to use a form, you can add the specific role/group label of interest to the slot mapping function
  ``from_entity`` (see :ref:`forms`).

  .. note::

      Composite entities are currently just supported by the :ref:`diet-classifier` and :ref:`CRFEntityExtractor`.
- `#5465 <https://github.com/rasahq/rasa/issues/5465>`_: Update training data format for NLU to support entities with a role or group label.

  You can now specify synonyms, roles, and groups of entities using the following data format:
  Markdown:

  .. code-block:: none

    [LA]{"entity": "location", "role": "city", "group": "CA", "value": "Los Angeles"}

  JSON:

  .. code-block:: none

    "entities": [
        {
            "start": 10,
            "end": 12,
            "value": "Los Angeles",
            "entity": "location",
            "role": "city",
            "group": "CA",
        }
    ]

  The markdown format ``[LA](location:Los Angeles)`` is deprecated. To update your training data file just
  execute the following command on the terminal of your choice:
  ``sed -i -E 's/\[([^)]+)\]\(([^)]+):([^)]+)\)/[\1]{"entity": "\2", "value": "\3"}/g' nlu.md``

  For more information about the new data format see :ref:`training-data-format`.

Improvements
------------
- `#2224 <https://github.com/rasahq/rasa/issues/2224>`_: Suppressed ``pika`` logs when establishing the connection. These log messages
  mostly happened when Rasa X and RabbitMQ were started at the same time. Since RabbitMQ
  can take a few seconds to initialize, Rasa X has to re-try until the connection is
  established.
  In case you suspect a different problem (such as failing authentication) you can
  re-enable the ``pika`` logs by setting the log level to ``DEBUG``. To run Rasa Open
  Source in debug mode, use the ``--debug`` flag. To run Rasa X in debug mode, set the
  environment variable ``DEBUG_MODE`` to ``true``.
- `#3419 <https://github.com/rasahq/rasa/issues/3419>`_: Include the source filename of a story in the failed stories

  Include the source filename of a story in the failed stories to make it easier to identify the file which contains the failed story.
- `#5544 <https://github.com/rasahq/rasa/issues/5544>`_: Add confusion matrix and "confused_with" to response selection evaluation

  If you are using ResponseSelectors, they now produce similiar outputs during NLU evaluation. Misclassfied responses are listed in a "confused_with" attribute in the evaluation report. Similiarily, a confusion matrix of all responses is plotted.
- `#5578 <https://github.com/rasahq/rasa/issues/5578>`_: Added ``socketio`` to the compatible channels for :ref:`reminders-and-external-events`.
- `#5595 <https://github.com/rasahq/rasa/issues/5595>`_: Update ``POST /model/train`` endpoint to accept retrieval action responses
  at the ``responses`` key of the JSON payload.
- `#5627 <https://github.com/rasahq/rasa/issues/5627>`_: All Rasa Open Source images are now using Python 3.7 instead of Python 3.6.
- `#5635 <https://github.com/rasahq/rasa/issues/5635>`_: Update dependencies based on the ``dependabot`` check.
- `#5636 <https://github.com/rasahq/rasa/issues/5636>`_: Add dropout between ``FFNN`` and ``DenseForSparse`` layers in ``DIETClassifier``,
  ``ResponseSelector`` and ``EmbeddingIntentClassifier`` controlled by ``use_dense_input_dropout`` config parameter.
- `#5646 <https://github.com/rasahq/rasa/issues/5646>`_: ``DIETClassifier`` only counts as extractor in ``rasa test`` if it was actually trained for entity recognition.
- `#5669 <https://github.com/rasahq/rasa/issues/5669>`_: Remove regularization gradient for variables that don't have prediction gradient.
- `#5672 <https://github.com/rasahq/rasa/issues/5672>`_: Raise a warning in ``CRFEntityExtractor`` and ``DIETClassifier`` if entities are not correctly annotated in the
  training data, e.g. their start and end values do not match any start and end values of tokens.
- `#5690 <https://github.com/rasahq/rasa/issues/5690>`_: Add ``full_retrieval_intent`` property to ``ResponseSelector`` rankings
- `#5717 <https://github.com/rasahq/rasa/issues/5717>`_: Change default values for hyper-parameters in ``EmbeddingIntentClassifier`` and ``DIETClassifier``

  Use ``scale_loss=False`` in ``DIETClassifier``. Reduce the number of dense dimensions for sparse features of text from 512 to 256 in ``EmbeddingIntentClassifier``.

Bugfixes
--------
- `#5230 <https://github.com/rasahq/rasa/issues/5230>`_: Fixed issue where posting to certain callback channel URLs would return a 500 error on successful posts due to invalid response format.
- `#5475 <https://github.com/rasahq/rasa/issues/5475>`_: One word can just have one entity label.

  If you are using, for example, ``ConveRTTokenizer`` words can be split into multiple tokens.
  Our entity extractors assign entity labels per token. So, it might happen, that a word, that was split into two tokens,
  got assigned two different entity labels. This is now fixed. One word can just have one entity label at a time.
- `#5509 <https://github.com/rasahq/rasa/issues/5509>`_: An entity label should always cover a complete word.

  If you are using, for example, ``ConveRTTokenizer`` words can be split into multiple tokens.
  Our entity extractors assign entity labels per token. So, it might happen, that just a part of a word has
  an entity label. This is now fixed. An entity label always covers a complete word.
- `#5574 <https://github.com/rasahq/rasa/issues/5574>`_: Fixed an issue that happened when metadata is passed in a new session.

  Now the metadata is correctly passed to the ActionSessionStart.
- `#5672 <https://github.com/rasahq/rasa/issues/5672>`_: Updated Python dependency ``ruamel.yaml`` to ``>=0.16``. We recommend to use at least
  ``0.16.10`` due to the security issue
  `CVE-2019-20478 <https://nvd.nist.gov/vuln/detail/CVE-2019-20478>`_ which is present in
  in prior versions.

Miscellaneous internal changes
------------------------------
- #5556, #5587, #5614, #5631, #5633


[1.9.7] - 2020-04-23
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- `#4606 <https://github.com/rasahq/rasa/issues/4606>`_: The stream reading timeout for ``rasa shell` is now configurable by using the
  environment variable ``RASA_SHELL_STREAM_READING_TIMEOUT_IN_SECONDS``.
  This can help to fix problems when using ``rasa shell`` with custom actions which run
  10 seconds or longer.

Bugfixes
--------
- `#5709 <https://github.com/rasahq/rasa/issues/5709>`_: Reverted changes in 1.9.6 that led to model incompatibility. Upgrade to 1.9.7 to fix 
  ``self.sequence_lengths_for(tf_batch_data[TEXT_SEQ_LENGTH][0]) IndexError: list index out of range`` 
  error without needing to retrain earlier 1.9 models.

  Therefore, all 1.9 models `except for 1.9.6` will be compatible; a model trained on 1.9.6 will need
  to be retrained on 1.9.7.


[1.9.6] - 2020-04-15
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5426 <https://github.com/rasahq/rasa/issues/5426>`_: Fix `rasa test nlu` plotting when using multiple runs.
- `#5489 <https://github.com/rasahq/rasa/issues/5489>`_: Fixed issue where ``max_number_of_predictions`` was not considered when running end-to-end testing.

Miscellaneous internal changes
------------------------------
- #5626


[1.9.5] - 2020-04-01
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- `#5533 <https://github.com/rasahq/rasa/issues/5533>`_: Support for
  `PostgreSQL schemas <https://www.postgresql.org/docs/11/ddl-schemas.html>`_ in
  :ref:`sql-tracker-store`. The ``SQLTrackerStore``
  accesses schemas defined by the ``POSTGRESQL_SCHEMA`` environment variable if
  connected to a PostgreSQL database.

  The schema is added to the connection string option's ``-csearch_path`` key, e.g.
  ``-options=-csearch_path=<SCHEMA_NAME>`` (see
  `<https://www.postgresql.org/docs/11/contrib-dblink-connect.html>`_ for more details).
  As before, if no ``POSTGRESQL_SCHEMA`` is defined, Rasa uses the database's default
  schema (``public``).

  The schema has to exist in the database before connecting, i.e. it needs to have been
  created with

  .. code-block:: postgresql

    CREATE SCHEMA schema_name;

Bugfixes
--------
- `#5547 <https://github.com/rasahq/rasa/issues/5547>`_: Fixed ambiguous logging in ``DIETClassifier`` by adding the name of the calling class to the log message.


[1.9.4] - 2020-03-30
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5529 <https://github.com/rasahq/rasa/issues/5529>`_: Fix memory leak problem on increasing number of calls to ``/model/parse`` endpoint.


[1.9.3] - 2020-03-27
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5505 <https://github.com/rasahq/rasa/issues/5505>`_: Set default value for ``weight_sparsity`` in ``ResponseSelector`` to ``0``.
  This fixes a bug in the default behaviour of ``ResponseSelector`` which was accidentally introduced in ``rasa==1.8.0``.
  Users should update to this version and re-train their models if ``ResponseSelector`` was used in their pipeline.


[1.9.2] - 2020-03-26
^^^^^^^^^^^^^^^^^^^^

Improved Documentation
----------------------
- `#5497 <https://github.com/RasaHQ/rasa/pull/5497>`_: Fix documentation to bring back Sara.


[1.9.1] - 2020-03-25
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5492 <https://github.com/rasahq/rasa/issues/5492>`_: Fix an issue where the deprecated ``queue`` parameter for the :ref:`event-brokers-pika`
  was ignored and Rasa Open Source published the events to the ``rasa_core_events``
  queue instead. Note that this does not change the fact that the ``queue`` argument
  is deprecated in favor of the ``queues`` argument.


[1.9.0] - 2020-03-24
^^^^^^^^^^^^^^^^^^^^

Features
--------
- `#5006 <https://github.com/rasahq/rasa/issues/5006>`_: Channel ``hangouts`` for Rasa integration with Google Hangouts Chat is now supported out-of-the-box.
- `#5389 <https://github.com/rasahq/rasa/issues/5389>`_: Add an optional path to a specific directory to download and cache the pre-trained model weights for :ref:`HFTransformersNLP`.
- `#5422 <https://github.com/rasahq/rasa/issues/5422>`_: Add options ``tensorboard_log_directory`` and ``tensorboard_log_level`` to ``EmbeddingIntentClassifier``,
  ``DIETClasifier``, ``ResponseSelector``, ``EmbeddingPolicy`` and ``TEDPolicy``.

  By default ``tensorboard_log_directory`` is ``None``. If a valid directory is provided,
  metrics are written during training. After the model is trained you can take a look
  at the training metrics in tensorboard. Execute ``tensorboard --logdir <path-to-given-directory>``.

  Metrics can either be written after every epoch (default) or for every training step.
  You can specify when to write metrics using the variable ``tensorboard_log_level``.
  Valid values are 'epoch' and 'minibatch'.

  We also write down a model summary, i.e. layers with inputs and types, to the given directory.

Improvements
------------
- `#4756 <https://github.com/rasahq/rasa/issues/4756>`_: Make response timeout configurable.
  ``rasa run``, ``rasa shell`` and ``rasa x`` can now be started with
  ``--response-timeout <int>`` to configure a response timeout of ``<int>`` seconds.
- `#4826 <https://github.com/rasahq/rasa/issues/4826>`_: Add full retrieval intent name to message data
  ``ResponseSelector`` will now add the full retrieval intent name
  e.g. ``faq/which_version`` to the prediction, making it accessible
  from the tracker.
- `#5258 <https://github.com/rasahq/rasa/issues/5258>`_: Added ``PikaEventBroker`` (:ref:`event-brokers-pika`) support for publishing to
  multiple queues. Messages are now published to a ``fanout`` exchange with name
  ``rasa-exchange`` (see
  `exchange-fanout <https://www.rabbitmq.com/tutorials/amqp-concepts.html#exchange-fanout>`_
  for more information on ``fanout`` exchanges).

  The former ``queue`` key is deprecated. Queues should now be
  specified as a list in the ``endpoints.yml`` event broker config under a new key
  ``queues``. Example config:

  .. code-block:: yaml

      event_broker:
        type: pika
        url: localhost
        username: username
        password: password
        queues:
          - queue-1
          - queue-2
          - queue-3
- `#5416 <https://github.com/rasahq/rasa/issues/5416>`_: Change ``rasa init`` to include ``tests/conversation_tests.md`` file by default.
- `#5446 <https://github.com/rasahq/rasa/issues/5446>`_: The endpoint ``PUT /conversations/<conversation_id>/tracker/events`` no longer
  adds session start events (to learn more about conversation sessions, please
  see :ref:`session_config`) in addition to the events which were sent in the request
  payload. To achieve the old behavior send a
  ``GET /conversations/<conversation_id>/tracker``
  request before appending events.
- `#5482 <https://github.com/rasahq/rasa/issues/5482>`_: Make ``scale_loss`` for intents behave the same way as in versions below ``1.8``, but
  only scale if some of the examples in a batch has probability of the golden label more than ``0.5``.
  Introduce ``scale_loss`` for entities in ``DIETClassifier``.

Bugfixes
--------
- `#5205 <https://github.com/rasahq/rasa/issues/5205>`_: Fixed the bug when FormPolicy was overwriting MappingPolicy prediction (e.g. ``/restart``).
  Priorities for :ref:`mapping-policy` and :ref:`form-policy` are no longer linear:
  ``FormPolicy`` priority is 5, but its prediction is ignored if ``MappingPolicy`` is used for prediction.
- `#5215 <https://github.com/rasahq/rasa/issues/5215>`_: Fixed issue related to storing Python ``float`` values as ``decimal.Decimal`` objects
  in DynamoDB tracker stores. All ``decimal.Decimal`` objects are now converted to
  ``float`` on tracker retrieval.

  Added a new docs section on :ref:`tracker-stores-dynamo`.
- `#5356 <https://github.com/rasahq/rasa/issues/5356>`_: Fixed bug where ``FallbackPolicy`` would always fall back if the fallback action is
  ``action_listen``.
- `#5361 <https://github.com/rasahq/rasa/issues/5361>`_: Fixed bug where starting or ending a response with ``\n\n`` led to one of the responses returned being empty.
- `#5405 <https://github.com/rasahq/rasa/issues/5405>`_: Fixes issue where model always gets retrained if multiple NLU/story files are in a 
  directory, by sorting the list of files.
- `#5444 <https://github.com/rasahq/rasa/issues/5444>`_: Fixed ambiguous logging in `DIETClassifier` by adding the name of the calling class to the log message.

Improved Documentation
----------------------
- `#2237 <https://github.com/rasahq/rasa/issues/2237>`_: Restructure the "Evaluating models" documentation page and rename this page to :ref:`testing-your-assistant`.
- `#5302 <https://github.com/rasahq/rasa/issues/5302>`_: Improved documentation on how to build and deploy an action server image for use on other servers such as Rasa X deployments.

Miscellaneous internal changes
------------------------------
- #5340


[1.8.3] - 2020-03-27
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5405 <https://github.com/rasahq/rasa/issues/5405>`_: Fixes issue where model always gets retrained if multiple NLU/story files are in a 
  directory, by sorting the list of files.
- `#5444 <https://github.com/rasahq/rasa/issues/5444>`_: Fixed ambiguous logging in `DIETClassifier` by adding the name of the calling class to the log message.
- `#5506 <https://github.com/rasahq/rasa/issues/5506>`_: Set default value for ``weight_sparsity`` in ``ResponseSelector`` to ``0``.
  This fixes a bug in the default behaviour of ``ResponseSelector`` which was accidentally introduced in ``rasa==1.8.0``.
  Users should update to this version or ``rasa>=1.9.3`` and re-train their models if ``ResponseSelector`` was used in their pipeline.

Improved Documentation
----------------------
- `#5302 <https://github.com/rasahq/rasa/issues/5302>`_: Improved documentation on how to build and deploy an action server image for use on other servers such as Rasa X deployments.


[1.8.2] - 2020-03-19
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5438 <https://github.com/rasahq/rasa/issues/5438>`_: Fixed bug when installing rasa with ``poetry``.
- `#5413 <https://github.com/RasaHQ/rasa/issues/5413>`_: Fixed bug with ``EmbeddingIntentClassifier``, where results
  weren't the same as in 1.7.x. Fixed by setting weight sparsity to 0.

Improved Documentation
----------------------
- `#5404 <https://github.com/rasahq/rasa/issues/5404>`_: Explain how to run commands as ``root`` user in Rasa SDK Docker images since version
  ``1.8.0``. Since version ``1.8.0`` the Rasa SDK Docker images does not longer run as
  ``root`` user by default. For commands which require ``root`` user usage, you have to
  switch back to the ``root`` user in your Docker image as described in
  :ref:`building-an-action-server-image`.
- `#5402 <https://github.com/RasaHQ/rasa/issues/5402>`_: Made improvements to Building Assistants tutorial


[1.8.1] - 2020-03-06
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5354 <https://github.com/rasahq/rasa/issues/5354>`_: Fixed issue with using language models like ``xlnet`` along with ``entity_recognition`` set to ``True`` inside
  ``DIETClassifier``.

Miscellaneous internal changes
------------------------------
- #5330, #5348


[1.8.0] - 2020-02-26
^^^^^^^^^^^^^^^^^^^^

Deprecations and Removals
-------------------------
- `#4991 <https://github.com/rasahq/rasa/issues/4991>`_: Removed ``Agent.continue_training`` and the ``dump_flattened_stories`` parameter
  from ``Agent.persist``.
- `#5266 <https://github.com/rasahq/rasa/issues/5266>`_: Properties ``Component.provides`` and ``Component.requires`` are deprecated.
  Use ``Component.required_components()`` instead.

Features
--------
- `#2674 <https://github.com/rasahq/rasa/issues/2674>`_: Add default value ``__other__`` to ``values`` of a ``CategoricalSlot``.

  All values not mentioned in the list of values of a ``CategoricalSlot``
  will be mapped to ``__other__`` for featurization.
- `#4088 <https://github.com/rasahq/rasa/issues/4088>`_: Add story structure validation functionality (e.g. `rasa data validate stories --max-history 5`).
- `#5065 <https://github.com/rasahq/rasa/issues/5065>`_: Add :ref:`LexicalSyntacticFeaturizer` to sparse featurizers.

  ``LexicalSyntacticFeaturizer`` does the same featurization as the ``CRFEntityExtractor``. We extracted the
  featurization into a separate component so that the features can be reused and featurization is independent from the
  entity extraction.
- `#5187 <https://github.com/rasahq/rasa/issues/5187>`_: Integrate language models from HuggingFace's `Transformers <https://github.com/huggingface/transformers>`_ Library.

  Add a new NLP component :ref:`HFTransformersNLP` which tokenizes and featurizes incoming messages using a specified
  pre-trained model with the Transformers library as the backend.
  Add :ref:`LanguageModelTokenizer` and :ref:`LanguageModelFeaturizer` which use the information from
  :ref:`HFTransformersNLP` and sets them correctly for message object.
  Language models currently supported: BERT, OpenAIGPT, GPT-2, XLNet, DistilBert, RoBERTa.
- `#5225 <https://github.com/rasahq/rasa/issues/5225>`_: Added a new CLI command ``rasa export`` to publish tracker events from a persistent
  tracker store using an event broker. See :ref:`section_export`, :ref:`tracker-stores`
  and :ref:`event-brokers` for more details.
- `#5230 <https://github.com/rasahq/rasa/issues/5230>`_: Refactor how GPU and CPU environments are configured for TensorFlow 2.0.

  Please refer to the :ref:`documentation <tensorflow_usage>` to understand
  which environment variables to set in what scenarios. A couple of examples are shown below as well:

  .. code-block:: python

      # This specifies to use 1024 MB of memory from GPU with logical ID 0 and 2048 MB of memory from GPU with logical ID 1
      TF_GPU_MEMORY_ALLOC="0:1024, 1:2048"

      # Specifies that at most 3 CPU threads can be used to parallelize multiple non-blocking operations
      TF_INTER_OP_PARALLELISM_THREADS="3"

      # Specifies that at most 2 CPU threads can be used to parallelize a particular operation.
      TF_INTRA_OP_PARALLELISM_THREADS="2"

- `#5266 <https://github.com/rasahq/rasa/issues/5266>`_: Added a new NLU component :ref:`DIETClassifier <diet-classifier>` and a new policy :ref:`TEDPolicy <ted_policy>`.

  DIET (Dual Intent and Entity Transformer) is a multi-task architecture for intent classification and entity
  recognition. You can read more about this component in our :ref:`documentation <diet-classifier>`.
  The new component will replace the ``EmbeddingIntentClassifier`` and the
  :ref:`CRFEntityExtractor` in the future.
  Those two components are deprecated from now on.
  See :ref:`migration guide <migration-to-rasa-1.8>` for details on how to
  switch to the new component.

  :ref:`TEDPolicy <ted_policy>` is the new name for :ref:`EmbeddingPolicy <embedding_policy>`.
  ``EmbeddingPolicy`` is deprecated from now on.
  The functionality of ``TEDPolicy`` and ``EmbeddingPolicy`` is the same.
  Please update your configuration file to use the new name for the policy.
- `#663 <https://github.com/rasahq/rasa/issues/663>`_: The sentence vector of the ``SpacyFeaturizer`` and ``MitieFeaturizer`` can be calculated using max or mean pooling.

  To specify the pooling operation, set the option ``pooling`` for the ``SpacyFeaturizer`` or the ``MitieFeaturizer``
  in your configuration file. The default pooling operation is ``mean``. The mean pooling operation also does not take
  into account words, that do not have a word vector.
  See our :ref:`documentation <components>` for more details.

Improvements
------------
- `#3975 <https://github.com/rasahq/rasa/issues/3975>`_: Added command line argument ``--conversation-id`` to ``rasa interactive``.
  If the argument is not given, ``conversation_id`` defaults to a random uuid.
- `#4653 <https://github.com/rasahq/rasa/issues/4653>`_: Added a new command-line argument ``--init-dir`` to command ``rasa init`` to specify
  the directory in which the project is initialised.
- `#4682 <https://github.com/rasahq/rasa/issues/4682>`_: Added support to send images with the twilio output channel.
- `#4817 <https://github.com/rasahq/rasa/issues/4817>`_: Part of Slack sanitization:
  Multiple garbled URL's in a string coming from slack will be converted into actual strings.
  ``Example: health check of <http://eemdb.net|eemdb.net> and <http://eemdb1.net|eemdb1.net> to health check of
  eemdb.net and eemdb1.net``
- `#5117 <https://github.com/rasahq/rasa/issues/5117>`_: New command-line argument --conversation-id will be added and wiil give the ability to
  set specific conversation ID for each shell session, if not passed will be random.
- `#5211 <https://github.com/rasahq/rasa/issues/5211>`_: Messages sent to the :ref:`event-brokers-pika` are now persisted. This guarantees
  the RabbitMQ will re-send previously received messages after a crash. Note that this
  does not help for the case where messages are sent to an unavailable RabbitMQ instance.
- `#5250 <https://github.com/rasahq/rasa/issues/5250>`_: Added support for mattermost connector to use bot accounts.
- `#5266 <https://github.com/rasahq/rasa/issues/5266>`_: We updated our code to TensorFlow 2.
- `#5317 <https://github.com/rasahq/rasa/issues/5317>`_: Events exported using ``rasa export`` receive a message header if published through a
  ``PikaEventBroker``. The header is added to the message's ``BasicProperties.headers``
  under the ``rasa-export-process-id`` key
  (``rasa.core.constants.RASA_EXPORT_PROCESS_ID_HEADER_NAME``). The value is a
  UUID4 generated at each call of ``rasa export``. The resulting header is a key-value
  pair that looks as follows:

  .. code-block:: text

    'rasa-export-process-id': 'd3b3d3ffe2bd4f379ccf21214ccfb261'

- `#5292 <https://github.com/rasahq/rasa/issues/5292>`_: Added ``followlinks=True`` to os.walk calls, to allow the use of symlinks in training, NLU and domain data.
- `#4811 <https://github.com/rasahq/rasa/issues/4811>`_: Support invoking a ``SlackBot`` by direct messaging or ``@<app name>`` mentions.

Bugfixes
--------
- `#4006 <https://github.com/rasahq/rasa/issues/4006>`_: Fixed timestamp parsing warning when using DucklingHTTPExtractor
- `#4601 <https://github.com/rasahq/rasa/issues/4601>`_: Fixed issue with ``action_restart`` getting overridden by ``action_listen`` when the ``MappingPolicy`` and the
  `TwoStageFallbackPolicy <https://rasa.com/docs/rasa/core/policies/#two-stage-fallback-policy>`_ are used together.
- `#5201 <https://github.com/rasahq/rasa/issues/5201>`_: Fixed incorrectly raised Error encountered in pipelines with a ``ResponseSelector`` and NLG.

  When NLU training data is split before NLU pipeline comparison,
  NLG responses were not also persisted and therefore training for a pipeline including the ``ResponseSelector`` would fail.

  NLG responses are now persisted along with NLU data to a ``/train`` directory in the ``run_x/xx%_exclusion`` folder.
- `#5277 <https://github.com/rasahq/rasa/issues/5277>`_: Fixed sending custom json with Twilio channel

Improved Documentation
----------------------
- `#5174 <https://github.com/rasahq/rasa/issues/5174>`_: Updated the documentation to properly suggest not to explicitly add utterance actions to the domain.
- `#5189 <https://github.com/rasahq/rasa/issues/5189>`_: Added user guide for reminders and external events, including ``reminderbot`` demo.

Miscellaneous internal changes
------------------------------
- #3923, #4597, #4903, #5180, #5189, #5266, #699


[1.7.4] - 2020-02-24
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5068 <https://github.com/rasahq/rasa/issues/5068>`_: Tracker stores supporting conversation sessions (``SQLTrackerStore`` and
  ``MongoTrackerStore``) do not save the tracker state to database immediately after
  starting a new conversation session. This leads to the number of events being saved
  in addition to the already-existing ones to be calculated correctly.

  This fixes ``action_listen`` events being saved twice at the beginning of
  conversation sessions.


[1.7.3] - 2020-02-21
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5231 <https://github.com/rasahq/rasa/issues/5231>`_: Fix segmentation fault when running ``rasa train`` or ``rasa shell``.

Improved Documentation
----------------------
- `#5286 <https://github.com/rasahq/rasa/issues/5286>`_: Fix doc links on "Deploying your Assistant" page


[1.7.2] - 2020-02-13
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5197 <https://github.com/rasahq/rasa/issues/5197>`_: Fixed incompatibility of Oracle with the :ref:`sql-tracker-store`, by using a ``Sequence``
  for the primary key columns. This does not change anything for SQL databases other than Oracle.
  If you are using Oracle, please create a sequence with the instructions in the :ref:`sql-tracker-store` docs.

Improved Documentation
----------------------
- `#5197 <https://github.com/rasahq/rasa/issues/5197>`_: Added section on setting up the SQLTrackerStore with Oracle
- `#5210 <https://github.com/rasahq/rasa/issues/5210>`_: Renamed "Running the Server" page to "Configuring the HTTP API"


[1.7.1] - 2020-02-11
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#5106 <https://github.com/rasahq/rasa/issues/5106>`_: Fixed file loading of non proper UTF-8 story files, failing properly when checking for
  story files.
- `#5162 <https://github.com/rasahq/rasa/issues/5162>`_: Fix problem with multi-intents.
  Training with multi-intents using the ``CountVectorsFeaturizer`` together with ``EmbeddingIntentClassifier`` is
  working again.
- `#5171 <https://github.com/rasahq/rasa/issues/5171>`_: Fix bug ``ValueError: Cannot concatenate sparse features as sequence dimension does not match``.

  When training a Rasa model that contains responses for just some of the intents, training was failing.
  Fixed the featurizers to return a consistent feature vector in case no response was given for a specific message.
- `#5199 <https://github.com/rasahq/rasa/issues/5199>`_: If no text features are present in ``EmbeddingIntentClassifier`` return the intent ``None``.
- `#5216 <https://github.com/rasahq/rasa/issues/5216>`_: Resolve version conflicts: Pin version of cloudpickle to ~=1.2.0.


[1.7.0] - 2020-01-29
^^^^^^^^^^^^^^^^^^^^

Deprecations and Removals
-------------------------
- `#4964 <https://github.com/rasahq/rasa/issues/4964>`_: The endpoint ``/conversations/<conversation_id>/execute`` is now deprecated. Instead, users should use
  the ``/conversations/<conversation_id>/trigger_intent`` endpoint and thus trigger intents instead of actions.
- `#4978 <https://github.com/rasahq/rasa/issues/4978>`_: Remove option ``use_cls_token`` from tokenizers and option ``return_sequence`` from featurizers.

  By default all tokenizer add a special token (``__CLS__``) to the end of the list of tokens.
  This token will be used to capture the features of the whole utterance.

  The featurizers will return a matrix of size (number-of-tokens x feature-dimension) by default.
  This allows to train sequence models.
  However, the feature vector of the ``__CLS__`` token can be used to train non-sequence models.
  The corresponding classifier can decide what kind of features to use.

Features
--------
- `#400 <https://github.com/rasahq/rasa/issues/400>`_: Rename ``templates`` key in domain to ``responses``.

  ``templates`` key will still work for backwards compatibility but will raise a future warning.
- `#4902 <https://github.com/rasahq/rasa/issues/4902>`_: Added a new configuration parameter, ``ranking_length`` to the ``EmbeddingPolicy``, ``EmbeddingIntentClassifier``,
  and ``ResponseSelector`` classes.
- `#4964 <https://github.com/rasahq/rasa/issues/4964>`_: External events and reminders now trigger intents (and entities) instead of actions.

  Add new endpoint ``/conversations/<conversation_id>/trigger_intent``, which lets the user specify an intent and a
  list of entities that is injected into the conversation in place of a user message. The bot then predicts and
  executes a response action.
- `#4978 <https://github.com/rasahq/rasa/issues/4978>`_: Add ``ConveRTTokenizer``.

  The tokenizer should be used whenever the ``ConveRTFeaturizer`` is used.

  Every tokenizer now supports the following configuration options:
  ``intent_tokenization_flag``: Flag to check whether to split intents (default ``False``).
  ``intent_split_symbol``: Symbol on which intent should be split (default ``_``)

Improvements
------------
- `#1988 <https://github.com/rasahq/rasa/issues/1988>`_: Remove the need of specifying utter actions in the ``actions`` section explicitly if these actions are already
  listed in the ``templates`` section.
- `#4877 <https://github.com/rasahq/rasa/issues/4877>`_: Entity examples that have been extracted using an external extractor are excluded
  from Markdown dumping in ``MarkdownWriter.dumps()``. The excluded external extractors
  are ``DucklingHTTPExtractor`` and ``SpacyEntityExtractor``.
- `#4902 <https://github.com/rasahq/rasa/issues/4902>`_: The ``EmbeddingPolicy``, ``EmbeddingIntentClassifier``, and ``ResponseSelector`` now by default normalize confidence
  levels over the top 10 results. See :ref:`migration-to-rasa-1.7` for more details.
- `#4964 <https://github.com/rasahq/rasa/issues/4964>`_: ``ReminderCancelled`` can now cancel multiple reminders if no name is given. It still cancels a single
  reminder if the reminder's name is specified.

Bugfixes
--------
- `#4774 <https://github.com/rasahq/rasa/issues/4774>`_: Requests to ``/model/train`` do not longer block other requests to the Rasa server.
- `#4896 <https://github.com/rasahq/rasa/issues/4896>`_: Fixed default behavior of ``rasa test core --evaluate-model-directory`` when called without ``--model``. Previously, the latest model file was used as ``--model``. Now the default model directory is used instead.

  New behavior of ``rasa test core --evaluate-model-directory`` when given an existing file as argument for ``--model``: Previously, this led to an error. Now a warning is displayed and the directory containing the given file is used as ``--model``.
- `#5040 <https://github.com/rasahq/rasa/issues/5040>`_: Updated the dependency ``networkx`` from 2.3.0 to 2.4.0. The old version created incompatibilities when using pip.

  There is an imcompatibility between Rasa dependecy requests 2.22.0 and the own depedency from Rasa for networkx raising errors upon pip install. There is also a bug corrected in ``requirements.txt`` which used ``~=`` instead of ``==``. All of these are fixed using networkx 2.4.0.
- `#5057 <https://github.com/rasahq/rasa/issues/5057>`_: Fixed compatibility issue with Microsoft Bot Framework Emulator if ``service_url`` lacked a trailing ``/``.
- `#5092 <https://github.com/rasahq/rasa/issues/5092>`_: DynamoDB tracker store decimal values will now be rounded on save. Previously values exceeding 38 digits caused an unhandled error.

Miscellaneous internal changes
------------------------------
- #4458, #4664, #4780, #5029


[1.6.2] - 2020-01-28
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- `#4994 <https://github.com/rasahq/rasa/issues/4994>`_: Switching back to a TensorFlow release which only includes CPU support to reduce the
  size of the dependencies. If you want to use the TensorFlow package with GPU support,
  please run ``pip install tensorflow-gpu==1.15.0``.

Bugfixes
--------
- `#5111 <https://github.com/rasahq/rasa/issues/5111>`_: Fixes ``Exception 'Loop' object has no attribute '_ready'`` error when running
  ``rasa init``.
- `#5126 <https://github.com/rasahq/rasa/issues/5126>`_: Updated the end-to-end ValueError you recieve when you have a invalid story format to point
  to the updated doc link.


[1.6.1] - 2020-01-07
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- `#4989 <https://github.com/rasahq/rasa/issues/4989>`_: Use an empty domain in case a model is loaded which has no domain
  (avoids errors when accessing ``agent.doman.<some attribute>``).
- `#4995 <https://github.com/rasahq/rasa/issues/4995>`_: Replace error message with warning in tokenizers and featurizers if default parameter not set.
- `#5019 <https://github.com/rasahq/rasa/issues/5019>`_: Pin sanic patch version instead of minor version. Fixes sanic ``_run_request_middleware()`` error.
- `#5032 <https://github.com/rasahq/rasa/issues/5032>`_: Fix wrong calculation of additional conversation events when saving the conversation.
  This led to conversation events not being saved.
- `#5032 <https://github.com/rasahq/rasa/issues/5032>`_: Fix wrong order of conversation events when pushing events to conversations via
  ``POST /conversations/<conversation_id>/tracker/events``.


[1.6.0] - 2019-12-18
^^^^^^^^^^^^^^^^^^^^

Deprecations and Removals
-------------------------
- `#4935 <https://github.com/rasahq/rasa/issues/4935>`_: Removed ``ner_features`` as a feature name from ``CRFEntityExtractor``, use ``text_dense_features`` instead.

  The following settings match the previous ``NGramFeaturizer``:

  .. code-block:: yaml

      - name: 'CountVectorsFeaturizer'
          analyzer: 'char_wb'
          min_ngram: 3
          max_ngram: 17
          max_features: 10
          min_df: 5
- `#4957 <https://github.com/rasahq/rasa/issues/4957>`_: To use custom features in the ``CRFEntityExtractor`` use ``text_dense_features`` instead of ``ner_features``. If
  ``text_dense_features`` are present in the feature set, the ``CRFEntityExtractor`` will automatically make use of
  them. Just make sure to add a dense featurizer in front of the ``CRFEntityExtractor`` in your pipeline and set the
  flag ``return_sequence`` to ``True`` for that featurizer.
  See https://rasa.com/docs/rasa/nlu/entity-extraction/#passing-custom-features-to-crfentityextractor.
- `#4990 <https://github.com/rasahq/rasa/issues/4990>`_: Deprecated ``Agent.continue_training``. Instead, a model should be retrained.
- `#684 <https://github.com/rasahq/rasa/issues/684>`_: Specifying lookup tables directly in the NLU file is now deprecated. Please specify
  them in an external file.

Features
--------
- `#4795 <https://github.com/rasahq/rasa/issues/4795>`_: Replaced the warnings about missing templates, intents etc. in validator.py by debug messages.
- `#4830 <https://github.com/rasahq/rasa/issues/4830>`_: Added conversation sessions to trackers.

  A conversation session represents the dialog between the assistant and a user.
  Conversation sessions can begin in three ways: 1. the user begins the conversation
  with the assistant, 2. the user sends their first message after a configurable period
  of inactivity, or 3. a manual session start is triggered with the ``/session_start``
  intent message. The period of inactivity after which a new conversation session is
  triggered is defined in the domain using the ``session_expiration_time`` key in the
  ``session_config`` section. The introduction of conversation sessions comprises the
  following changes:

  - Added a new event ``SessionStarted`` that marks the beginning of a new conversation
    session.
  - Added a new default action ``ActionSessionStart``. This action takes all
    ``SlotSet`` events from the previous session and applies it to the next session.
  - Added a new default intent ``session_start`` which triggers the start of a new
    conversation session.
  - ``SQLTrackerStore`` and ``MongoTrackerStore`` only retrieve
    events from the last session from the database.


  .. note::

    The session behaviour is disabled for existing projects, i.e. existing domains
    without session config section.
- `#4935 <https://github.com/rasahq/rasa/issues/4935>`_: Preparation for an upcoming change in the ``EmbeddingIntentClassifier``:

  Add option ``use_cls_token`` to all tokenizers. If it is set to ``True``, the token ``__CLS__`` will be added to
  the end of the list of tokens. Default is set to ``False``. No need to change the default value for now.

  Add option ``return_sequence`` to all featurizers. By default all featurizers return a matrix of size
  (1 x feature-dimension). If the option ``return_sequence`` is set to ``True``, the corresponding featurizer will return
  a matrix of size (token-length x feature-dimension). See https://rasa.com/docs/rasa/nlu/components/#featurizers.
  Default value is set to ``False``. However, you might want to set it to ``True`` if you want to use custom features
  in the ``CRFEntityExtractor``.
  See https://rasa.com/docs/rasa/nlu/entity-extraction/#passing-custom-features-to-crfentityextractor.

  Changed some featurizers to use sparse features, which should reduce memory usage with large amounts of training data significantly.
  Read more: :ref:`text-featurizers` .

  .. warning::

      These changes break model compatibility. You will need to retrain your old models!

Improvements
------------
- `#3549 <https://github.com/rasahq/rasa/issues/3549>`_: Added ``--no-plot`` option for ``rasa test`` command, which disables rendering of confusion matrix and histogram. By default plots will be rendered.
- `#4086 <https://github.com/rasahq/rasa/issues/4086>`_: If matplotlib couldn't set up a default backend, it will be set automatically to TkAgg/Agg one
- `#4647 <https://github.com/rasahq/rasa/issues/4647>`_: Add the option ```random_seed``` to the ```rasa data split nlu``` command to generate
  reproducible train/test splits.
- `#4734 <https://github.com/rasahq/rasa/issues/4734>`_: Changed ``url`` ``__init__()`` arguments for custom tracker stores to ``host`` to reflect the ``__init__`` arguments of
  currently supported tracker stores. Note that in ``endpoints.yml``, these are still declared as ``url``.
- `#4751 <https://github.com/rasahq/rasa/issues/4751>`_: The ``kafka-python`` dependency has become as an "extra" dependency. To use the
  ``KafkaEventConsumer``, ``rasa`` has to be installed with the ``[kafka]`` option, i.e.

  .. code-block:: bash

    $ pip install rasa[kafka]
- `#4801 <https://github.com/rasahq/rasa/issues/4801>`_: Allow creation of natural language interpreter and generator by classname reference
  in ``endpoints.yml``.
- `#4834 <https://github.com/rasahq/rasa/issues/4834>`_: Made it explicit that interactive learning does not work with NLU-only models.

  Interactive learning no longer trains NLU-only models if no model is provided
  and no core data is provided.
- `#4899 <https://github.com/rasahq/rasa/issues/4899>`_: The ``intent_report.json`` created by ``rasa test`` now creates an extra field
  ``confused_with`` for each intent. This is a dictionary containing the names of
  the most common false positives when this intent should be predicted, and the
  number of such false positives.
- `#4976 <https://github.com/rasahq/rasa/issues/4976>`_: ``rasa test nlu --cross-validation`` now also includes an evaluation of the response selector.
  As a result, the train and test F1-score, accuracy and precision is logged for the response selector.
  A report is also generated in the ``results`` folder by the name ``response_selection_report.json``

Bugfixes
--------
- `#4635 <https://github.com/rasahq/rasa/issues/4635>`_: If a ``wait_time_between_pulls`` is configured for the model server in ``endpoints.yml``,
  this will be used instead of the default one when running Rasa X.
- `#4759 <https://github.com/rasahq/rasa/issues/4759>`_: Training Luis data with ``luis_schema_version`` higher than 4.x.x will show a warning instead of throwing an exception.
- `#4799 <https://github.com/rasahq/rasa/issues/4799>`_: Running ``rasa interactive`` with no NLU data now works, with the functionality of ``rasa interactive core``.
- `#4917 <https://github.com/rasahq/rasa/issues/4917>`_: When loading models from S3, namespaces (folders within a bucket) are now respected.
  Previously, this would result in an error upon loading the model.
- `#4925 <https://github.com/rasahq/rasa/issues/4925>`_: "rasa init" will ask if user wants to train a model
- `#4942 <https://github.com/rasahq/rasa/issues/4942>`_: Pin ``multidict`` dependency to 4.6.1 to prevent sanic from breaking,
  see https://github.com/huge-success/sanic/issues/1729
- `#4985 <https://github.com/rasahq/rasa/issues/4985>`_: Fix errors during training and testing of ``ResponseSelector``.


[1.5.3] - 2019-12-11
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- `#4933 <https://github.com/rasahq/rasa/issues/4933>`_: Improved error message that appears when an incorrect parameter is passed to a policy.

Bugfixes
--------
- `#4914 <https://github.com/rasahq/rasa/issues/4914>`_: Added ``rasa/nlu/schemas/config.yml`` to wheel package
- `#4942 <https://github.com/rasahq/rasa/issues/4942>`_: Pin ``multidict`` dependency to 4.6.1 to prevent sanic from breaking,
  see https://github.com/huge-success/sanic/issues/1729


[1.5.2] - 2019-12-09
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- `#3684 <https://github.com/rasahq/rasa/issues/3684>`_: ``rasa interactive`` will skip the story visualization of training stories in case
  there are more than 200 stories. Stories created during interactive learning will be
  visualized as before.
- `#4792 <https://github.com/rasahq/rasa/issues/4792>`_: The log level for SocketIO loggers, including ``websockets.protocol``, ``engineio.server``,
  and ``socketio.server``, is now handled by the ``LOG_LEVEL_LIBRARIES`` environment variable,
  where the default log level is ``ERROR``.
- `#4873 <https://github.com/rasahq/rasa/issues/4873>`_: Updated all example bots and documentation to use the updated ``dispatcher.utter_message()`` method from `rasa-sdk==1.5.0`.

Bugfixes
--------
- `#3684 <https://github.com/rasahq/rasa/issues/3684>`_: ``rasa interactive`` will not load training stories in case the visualization is
  skipped.
- `#4789 <https://github.com/rasahq/rasa/issues/4789>`_: Fixed error where spacy models where not found in the docker images.
- `#4802 <https://github.com/rasahq/rasa/issues/4802>`_: Fixed unnecessary ``kwargs`` unpacking in ``rasa.test.test_core`` call in ``rasa.test.test`` function.
- `#4898 <https://github.com/rasahq/rasa/issues/4898>`_: Training data files now get loaded in the same order (especially relevant to subdirectories) each time to ensure training consistency when using a random seed.
- `#4918 <https://github.com/rasahq/rasa/issues/4918>`_: Locks for tickets in ``LockStore`` are immediately issued without a redundant
  check for their availability.

Improved Documentation
----------------------
- `#4844 <https://github.com/rasahq/rasa/issues/4844>`_: Added ``towncrier`` to automatically collect changelog entries.
- `#4869 <https://github.com/rasahq/rasa/issues/4869>`_: Document the pipeline for ``pretrained_embeddings_convert`` in the pre-configured pipelines section.
- `#4894 <https://github.com/rasahq/rasa/issues/4894>`_: ``Proactively Reaching Out to the User Using Actions`` now correctly links to the
  endpoint specification.


[1.5.1] - 2019-11-27
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- When NLU training data is dumped as Markdown file the intents are not longer ordered
  alphabetically, but in the original order of given training data

Bugfixes
--------
- End to end stories now support literal payloads which specify entities, e.g.
  ``greet: /greet{"name": "John"}``
- Slots will be correctly interpolated if there are lists in custom response templates.
- Fixed compatibility issues with ``rasa-sdk`` ``1.5``
- Updated ``/status`` endpoint to show correct path to model archive

[1.5.0] - 2019-11-26
^^^^^^^^^^^^^^^^^^^^

Features
--------
- Added data validator that checks if domain object returned is empty. If so, exit early
  from the command ``rasa data validate``.
- Added the KeywordIntentClassifier.
- Added documentation for ``AugmentedMemoizationPolicy``.
- Fall back to ``InMemoryTrackerStore`` in case there is any problem with the current
  tracker store.
- Arbitrary metadata can now be attached to any ``Event`` subclass. The data must be
  stored under the ``metadata`` key when reading the event from a JSON object or
  dictionary.
- Add command line argument ``rasa x --config CONFIG``, to specify path to the policy
  and NLU pipeline configuration of your bot (default: ``config.yml``).
- Added a new NLU featurizer - ``ConveRTFeaturizer`` based on `ConveRT
  <https://github.com/PolyAI-LDN/polyai-models>`_ model released by PolyAI.
- Added a new preconfigured pipeline - ``pretrained_embeddings_convert``.

Improvements
------------
- Do not retrain the entire Core model if only the ``templates`` section of the domain
  is changed.
- Upgraded ``jsonschema`` version.

Deprecations and Removals
-------------------------
- Remove duplicate messages when creating training data (issues/1446).

Bugfixes
--------
- ``MultiProjectImporter`` now imports files in the order of the import statements
- Fixed server hanging forever on leaving ``rasa shell`` before first message
- Fixed rasa init showing traceback error when user does Keyboard Interrupt before choosing a project path
- ``CountVectorsFeaturizer`` featurizes intents only if its analyzer is set to ``word``
- Fixed bug where facebooks generic template was not rendered when buttons were ``None``
- Fixed default intents unnecessarily raising undefined parsing error

[1.4.6] - 2019-11-22
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- Fixed Rasa X not working when any tracker store was configured for Rasa.
- Use the matplotlib backend ``agg`` in case the ``tkinter`` package is not installed.

[1.4.5] - 2019-11-14
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- NLU-only models no longer throw warnings about parsing features not defined in the domain
- Fixed bug that stopped Dockerfiles from building version 1.4.4.
- Fixed format guessing for e2e stories with intent restated as ``/intent``

[1.4.4] - 2019-11-13
^^^^^^^^^^^^^^^^^^^^

Features
--------
- ``PikaEventProducer`` adds the RabbitMQ ``App ID`` message property to published
  messages with the value of the ``RASA_ENVIRONMENT`` environment variable. The
  message property will not be assigned if this environment variable isn't set.

Improvements
------------
- Updated Mattermost connector documentation to be more clear.
- Updated format strings to f-strings where appropriate.
- Updated tensorflow requirement to ``1.15.0``
- Dump domain using UTF-8 (to avoid ``\UXXXX`` sequences in the dumped files)

Bugfixes
--------
- Fixed exporting NLU training data in ``json`` format from ``rasa interactive``
- Fixed numpy deprecation warnings

[1.4.3] - 2019-10-29
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- Fixed ``Connection reset by peer`` errors and bot response delays when using the
  RabbitMQ event broker.

[1.4.2] - 2019-10-28
^^^^^^^^^^^^^^^^^^^^

Deprecations and Removals
-------------------------
- TensorFlow deprecation warnings are no longer shown when running ``rasa x``

Bugfixes
--------
- Fixed ``'Namespace' object has no attribute 'persist_nlu_data'`` error during
  interactive learning
- Pinned `networkx~=2.3.0` to fix visualization in `rasa interactive` and Rasa X
- Fixed ``No model found`` error when using ``rasa run actions`` with "actions"
  as a directory.

[1.4.1] - 2019-10-22
^^^^^^^^^^^^^^^^^^^^
Regression: changes from ``1.2.12`` were missing from ``1.4.0``, readded them

[1.4.0] - 2019-10-19
^^^^^^^^^^^^^^^^^^^^

Features
--------
- add flag to CLI to persist NLU training data if needed
- log a warning if the ``Interpreter`` picks up an intent or an entity that does not
  exist in the domain file.
- added ``DynamoTrackerStore`` to support persistence of agents running on AWS
- added docstrings for ``TrackerStore`` classes
- added buttons and images to mattermost.
- ``CRFEntityExtractor`` updated to accept arbitrary token-level features like word
  vectors (issues/4214)
- ``SpacyFeaturizer`` updated to add ``ner_features`` for ``CRFEntityExtractor``
- Sanitizing incoming messages from slack to remove slack formatting like ``<mailto:xyz@rasa.com|xyz@rasa.com>``
  or ``<http://url.com|url.com>`` and substitute it with original content
- Added the ability to configure the number of Sanic worker processes in the HTTP
  server (``rasa.server``) and input channel server
  (``rasa.core.agent.handle_channels()``). The number of workers can be set using the
  environment variable ``SANIC_WORKERS`` (default: 1). A value of >1 is allowed only in
  combination with ``RedisLockStore`` as the lock store.
- Botframework channel can handle uploaded files in ``UserMessage`` metadata.
- Added data validator that checks there is no duplicated example data across multiples intents

Improvements
------------
- Unknown sections in markdown format (NLU data) are not ignored anymore, but instead an error is raised.
- It is now easier to add metadata to a ``UserMessage`` in existing channels.
  You can do so by overwriting the method ``get_metadata``. The return value of this
  method will be passed to the ``UserMessage`` object.
- Tests can now be run in parallel
- Serialise ``DialogueStateTracker`` as json instead of pickle. **DEPRECATION warning**:
  Deserialisation of pickled trackers will be deprecated in version 2.0. For now,
  trackers are still loaded from pickle but will be dumped as json in any subsequent
  save operations.
- Event brokers are now also passed to custom tracker stores (using the ``event_broker`` parameter)
- Don't run the Rasa Docker image as ``root``.
- Use multi-stage builds to reduce the size of the Rasa Docker image.
- Updated the ``/status`` api route to use the actual model file location instead of the ``tmp`` location.

Deprecations and Removals
-------------------------
- **Removed Python 3.5 support**

Bugfixes
--------
- fixed missing ``tkinter`` dependency for running tests on Ubuntu
- fixed issue with ``conversation`` JSON serialization
- fixed the hanging HTTP call with ``ner_duckling_http`` pipeline
- fixed Interactive Learning intent payload messages saving in nlu files
- fixed DucklingHTTPExtractor dimensions by actually applying to the request


[1.3.10] - 2019-10-18
^^^^^^^^^^^^^^^^^^^^^

Features
--------
- Can now pass a package as an argument to the ``--actions`` parameter of the
  ``rasa run actions`` command.

Bugfixes
--------
- Fixed visualization of stories with entities which led to a failing
  visualization in Rasa X

[1.3.9] - 2019-10-10
^^^^^^^^^^^^^^^^^^^^

Features
--------
- Port of 1.2.10 (support for RabbitMQ TLS authentication and ``port`` key in
  event broker endpoint config).
- Port of 1.2.11 (support for passing a CA file for SSL certificate verification via the
  --ssl-ca-file flag).

Bugfixes
--------
- Fixed the hanging HTTP call with ``ner_duckling_http`` pipeline.
- Fixed text processing of ``intent`` attribute inside ``CountVectorFeaturizer``.
- Fixed ``argument of type 'NoneType' is not iterable`` when using ``rasa shell``,
  ``rasa interactive`` / ``rasa run``

[1.3.8] - 2019-10-08
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- Policies now only get imported if they are actually used. This removes
  TensorFlow warnings when starting Rasa X

Bugfixes
--------
- Fixed error ``Object of type 'MaxHistoryTrackerFeaturizer' is not JSON serializable``
  when running ``rasa train core``
- Default channel ``send_`` methods no longer support kwargs as they caused issues in incompatible channels

[1.3.7] - 2019-09-27
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- re-added TLS, SRV dependencies for PyMongo
- socketio can now be run without turning on the ``--enable-api`` flag
- MappingPolicy no longer fails when the latest action doesn't have a policy

[1.3.6] - 2019-09-21
^^^^^^^^^^^^^^^^^^^^

Features
--------
- Added the ability for users to specify a conversation id to send a message to when
  using the ``RasaChat`` input channel.

[1.3.5] - 2019-09-20
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- Fixed issue where ``rasa init`` would fail without spaCy being installed

[1.3.4] - 2019-09-20
^^^^^^^^^^^^^^^^^^^^

Features
--------
- Added the ability to set the ``backlog`` parameter in Sanics ``run()`` method using
  the ``SANIC_BACKLOG`` environment variable. This parameter sets the
  number of unaccepted connections the server allows before refusing new
  connections. A default value of 100 is used if the variable is not set.
- Status endpoint (``/status``) now also returns the number of training processes currently running

Bugfixes
--------
- Added the ability to properly deal with spaCy ``Doc``-objects created on
  empty strings as discussed `here <https://github.com/RasaHQ/rasa/issues/4445>`_.
  Only training samples that actually bear content are sent to ``self.nlp.pipe``
  for every given attribute. Non-content-bearing samples are converted to empty
  ``Doc``-objects. The resulting lists are merged with their preserved order and
  properly returned.
- asyncio warnings are now only printed if the callback takes more than 100ms
  (up from 1ms).
- ``agent.load_model_from_server`` no longer affects logging.

Improvements
------------
- The endpoint ``POST /model/train`` no longer supports specifying an output directory
  for the trained model using the field ``out``. Instead you can choose whether you
  want to save the trained model in the default model directory (``models``)
  (default behavior) or in a temporary directory by specifying the
  ``save_to_default_model_directory`` field in the training request.

[1.3.3] - 2019-09-13
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- Added a check to avoid training ``CountVectorizer`` for a particular
  attribute of a message if no text is provided for that attribute across
  the training data.
- Default one-hot representation for label featurization inside ``EmbeddingIntentClassifier`` if label features don't exist.
- Policy ensemble no longer incorrectly wrings "missing mapping policy" when
  mapping policy is present.
- "text" from ``utter_custom_json`` now correctly saved to tracker when using telegram channel

Deprecations and Removals
-------------------------
- Removed computation of ``intent_spacy_doc``. As a result, none of the spacy components process intents now.

[1.3.2] - 2019-09-10
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- SQL tracker events are retrieved ordered by timestamps. This fixes interactive
  learning events being shown in the wrong order.

[1.3.1] - 2019-09-09
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- Pin gast to == 0.2.2

[1.3.0] - 2019-09-05
^^^^^^^^^^^^^^^^^^^^

Features
--------
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

Improvements
------------
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

Bugfixes
--------
- ``rasa test nlu`` with a folder of configuration files
- ``MappingPolicy`` standard featurizer is set to ``None``
- Removed ``text`` parameter from send_attachment function in slack.py to avoid duplication of text output to slackbot
- server ``/status`` endpoint reports status when an NLU-only model is loaded

Deprecations and Removals
-------------------------
- Removed ``--report`` argument from ``rasa test nlu``. All output files are stored in the ``--out`` directory.

[1.2.12] - 2019-10-16
^^^^^^^^^^^^^^^^^^^^^

Features
--------
- Support for transit encryption with Redis via ``use_ssl: True`` in the tracker store config in endpoints.yml

[1.2.11] - 2019-10-09
^^^^^^^^^^^^^^^^^^^^^

Features
--------
- Support for passing a CA file for SSL certificate verification via the
  --ssl-ca-file flag

[1.2.10] - 2019-10-08
^^^^^^^^^^^^^^^^^^^^^

Features
--------
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

Bugfixes
--------
- Correctly pass SSL flag values to x CLI command (backport of


[1.2.8] - 2019-09-10
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- SQL tracker events are retrieved ordered by timestamps. This fixes interactive
  learning events being shown in the wrong order. Backport of ``1.3.2`` patch
  (PR #4427).


[1.2.7] - 2019-09-02
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- Added ``query`` dictionary argument to ``SQLTrackerStore`` which will be appended
  to the SQL connection URL as query parameters.


[1.2.6] - 2019-09-02
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- fixed bug that occurred when sending template ``elements`` through a channel that doesn't support them

[1.2.5] - 2019-08-26
^^^^^^^^^^^^^^^^^^^^

Features
--------
- SSL support for ``rasa run`` command. Certificate can be specified using
  ``--ssl-certificate`` and ``--ssl-keyfile``.

Bugfixes
--------
- made default augmentation value consistent across repo
- ``'/restart'`` will now also restart the bot if the tracker is paused


[1.2.4] - 2019-08-23
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- the ``SocketIO`` input channel now allows accesses from other origins
  (fixes ``SocketIO`` channel on Rasa X)

[1.2.3] - 2019-08-15
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- messages with multiple entities are now handled properly with e2e evaluation
- ``data/test_evaluations/end_to_end_story.md`` was re-written in the
  restaurantbot domain

[1.2.3] - 2019-08-15
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- messages with multiple entities are now handled properly with e2e evaluation
- ``data/test_evaluations/end_to_end_story.md`` was re-written in the restaurantbot domain

Bugfixes
--------
- Free text input was not allowed in the Rasa shell when the response template
  contained buttons, which has now been fixed.

[1.2.2] - 2019-08-07
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- ``UserUttered`` events always got the same timestamp

[1.2.1] - 2019-08-06
^^^^^^^^^^^^^^^^^^^^

Features
--------
- Docs now have an ``EDIT THIS PAGE`` button

Bugfixes
--------
- ``Flood control exceeded`` error in Telegram connector which happened because the
  webhook was set twice

[1.2.0] - 2019-08-01
^^^^^^^^^^^^^^^^^^^^

Features
--------
- add root route to server started without ``--enable-api`` parameter
- add ``--evaluate-model-directory`` to ``rasa test core`` to evaluate models
  from ``rasa train core -c <config-1> <config-2>``
- option to send messages to the user by calling
  ``POST /conversations/{conversation_id}/execute``

Improvements
------------
- ``Agent.update_model()`` and ``Agent.handle_message()`` now work without needing to set a domain
  or a policy ensemble
- Update pytype to ``2019.7.11``
- new event broker class: ``SQLProducer``. This event broker is now used when running locally with
  Rasa X
- API requests are not longer logged to ``rasa_core.log`` by default in order to avoid
  problems when running on OpenShift (use ``--log-file rasa_core.log`` to retain the
  old behavior)
- ``metadata`` attribute added to ``UserMessage``

Bugfixes
--------
- ``rasa test core`` can handle compressed model files
- rasa can handle story files containing multi line comments
- template will retain `{` if escaped with `{`. e.g. `{{"foo": {bar}}}` will result in `{"foo": "replaced value"}`

[1.1.8] - 2019-07-25
^^^^^^^^^^^^^^^^^^^^

Features
--------
- ``TrainingFileImporter`` interface to support customizing the process of loading
  training data
- fill slots for custom templates

Improvements
------------
- ``Agent.update_model()`` and ``Agent.handle_message()`` now work without needing to set a domain
  or a policy ensemble
- update pytype to ``2019.7.11``

Bugfixes
--------
- interactive learning bug where reverted user utterances were dumped to training data
- added timeout to terminal input channel to avoid freezing input in case of server
  errors
- fill slots for image, buttons, quick_replies and attachments in templates
- ``rasa train core`` in comparison mode stores the model files compressed (``tar.gz`` files)
- slot setting in interactive learning with the TwoStageFallbackPolicy


[1.1.7] - 2019-07-18
^^^^^^^^^^^^^^^^^^^^

Features
--------
- added optional pymongo dependencies ``[tls, srv]`` to ``requirements.txt`` for better mongodb support
- ``case_sensitive`` option added to ``WhiteSpaceTokenizer`` with ``true`` as default.

Bugfixes
--------
- validation no longer throws an error during interactive learning
- fixed wrong cleaning of ``use_entities`` in case it was a list and not ``True``
- updated the server endpoint ``/model/parse`` to handle also messages with the intent prefix
- fixed bug where "No model found" message appeared after successfully running the bot
- debug logs now print to ``rasa_core.log`` when running ``rasa x -vv`` or ``rasa run -vv``

[1.1.6] - 2019-07-12
^^^^^^^^^^^^^^^^^^^^

Features
--------
- rest channel supports setting a message's input_channel through a field
  ``input_channel`` in the request body

Improvements
------------
- recommended syntax for empty ``use_entities`` and ``ignore_entities`` in the domain file
  has been updated from ``False`` or ``None`` to an empty list (``[]``)

Bugfixes
--------
- ``rasa run`` without ``--enable-api`` does not require a local model anymore
- using ``rasa run`` with ``--enable-api`` to run a server now prints
  "running Rasa server" instead of "running Rasa Core server"
- actions, intents, and utterances created in ``rasa interactive`` can no longer be empty


[1.1.5] - 2019-07-10
^^^^^^^^^^^^^^^^^^^^

Features
--------
- debug logging now tells you which tracker store is connected
- the response of ``/model/train`` now includes a response header for the trained model filename
- ``Validator`` class to help developing by checking if the files have any errors
- project's code is now linted using flake8
- ``info`` log when credentials were provided for multiple channels and channel in
  ``--connector`` argument was specified at the same time
- validate export paths in interactive learning

Improvements
------------
- deprecate ``rasa.core.agent.handle_channels(...)`. Please use ``rasa.run(...)``
  or ``rasa.core.run.configure_app`` instead.
- ``Agent.load()`` also accepts ``tar.gz`` model file

Deprecations and Removals
-------------------------
- revert the stripping of trailing slashes in endpoint URLs since this can lead to
  problems in case the trailing slash is actually wanted
- starter packs were removed from Github and are therefore no longer tested by Travis script

Bugfixes
--------
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

Features
--------
- unfeaturize single entities
- added agent readiness check to the ``/status`` resource

Improvements
------------
- removed leading underscore from name of '_create_initial_project' function.

Bugfixes
--------
- fixed bug where facebook quick replies were not rendering
- take FB quick reply payload rather than text as input
- fixed bug where `training_data` path in `metadata.json` was an absolute path

[1.1.3] - 2019-06-14
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- fixed any inconsistent type annotations in code and some bugs revealed by
  type checker

[1.1.2] - 2019-06-13
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- fixed duplicate events appearing in tracker when using a PostgreSQL tracker store

[1.1.1] - 2019-06-13
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- fixed compatibility with Rasa SDK
- bot responses can contain ``custom`` messages besides other message types

[1.1.0] - 2019-06-13
^^^^^^^^^^^^^^^^^^^^

Features
--------
- nlu configs can now be directly compared for performance on a dataset
  in ``rasa test nlu``

Improvements
------------
- update the tracker in interactive learning through reverting and appending events
  instead of replacing the tracker
- ``POST /conversations/{conversation_id}/tracker/events`` supports a list of events

Bugfixes
--------
- fixed creation of ``RasaNLUHttpInterpreter``
- form actions are included in domain warnings
- default actions, which are overriden by custom actions and are listed in the
  domain are excluded from domain warnings
- SQL ``data`` column type to ``Text`` for compatibility with MySQL
- non-featurizer training parameters don't break `SklearnPolicy` anymore

[1.0.9] - 2019-06-10
^^^^^^^^^^^^^^^^^^^^

Improvements
------------
- revert PR #3739 (as this is a breaking change): set ``PikaProducer`` and
  ``KafkaProducer`` default queues back to ``rasa_core_events``

[1.0.8] - 2019-06-10
^^^^^^^^^^^^^^^^^^^^

Features
--------
- support for specifying full database urls in the ``SQLTrackerStore`` configuration
- maximum number of predictions can be set via the environment variable
  ``MAX_NUMBER_OF_PREDICTIONS`` (default is 10)

Improvements
------------
- default ``PikaProducer`` and ``KafkaProducer`` queues to ``rasa_production_events``
- exclude unfeaturized slots from domain warnings

Bugfixes
--------
- loading of additional training data with the ``SkillSelector``
- strip trailing slashes in endpoint URLs

[1.0.7] - 2019-06-06
^^^^^^^^^^^^^^^^^^^^

Features
--------
- added argument ``--rasa-x-port`` to specify the port of Rasa X when running Rasa X locally via ``rasa x``

Bugfixes
--------
- slack notifications from bots correctly render text
- fixed usage of ``--log-file`` argument for ``rasa run`` and ``rasa shell``
- check if correct tracker store is configured in local mode

[1.0.6] - 2019-06-03
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- fixed backwards incompatible utils changes

[1.0.5] - 2019-06-03
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- fixed spacy being a required dependency (regression)

[1.0.4] - 2019-06-03
^^^^^^^^^^^^^^^^^^^^

Features
--------
- automatic creation of index on the ``sender_id`` column when using an SQL
  tracker store. If you have an existing data and you are running into performance
  issues, please make sure to add an index manually using
  ``CREATE INDEX event_idx_sender_id ON events (sender_id);``.

Improvements
------------
- NLU evaluation in cross-validation mode now also provides intent/entity reports,
  confusion matrix, etc.

[1.0.3] - 2019-05-30
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- non-ascii characters render correctly in stories generated from interactive learning
- validate domain file before usage, e.g. print proper error messages if domain file
  is invalid instead of raising errors

[1.0.2] - 2019-05-29
^^^^^^^^^^^^^^^^^^^^

Features
--------
- added ``domain_warnings()`` method to ``Domain`` which returns a dict containing the
  diff between supplied {actions, intents, entities, slots} and what's contained in the
  domain

Bugfixes
--------
- fix lookup table files failed to load issues/3622
- buttons can now be properly selected during cmdline chat or when in interactive learning
- set slots correctly when events are added through the API
- mapping policy no longer ignores NLU threshold
- mapping policy priority is correctly persisted


[1.0.1] - 2019-05-21
^^^^^^^^^^^^^^^^^^^^

Bugfixes
--------
- updated installation command in docs for Rasa X

[1.0.0] - 2019-05-21
^^^^^^^^^^^^^^^^^^^^

Features
--------
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

Improvements
------------
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

Deprecations and Removals
-------------------------
- removed possibility to execute ``python -m rasa_core.train`` etc. (e.g. scripts in ``rasa.core`` and ``rasa.nlu``).
  Use the CLI for rasa instead, e.g. ``rasa train core``.
- removed ``_sklearn_numpy_warning_fix`` from the ``SklearnIntentClassifier``
- removed ``Dispatcher`` class from core
- removed projects: the Rasa NLU server now has a maximum of one model at a time loaded.

Bugfixes
--------
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

.. _`Semantic Versioning`: https://semver.org/
