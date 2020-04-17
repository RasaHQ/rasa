:desc: Rasa Core Changelog

.. _old-core-change-log:

Core Change Log
===============

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 0.2.0.

[0.14.4] - 2019-05-13
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- correctly process form actions in core evaluations

[0.14.3] - 2019-05-07
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed interactive learning history printing

[0.14.2] - 2019-05-07
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed required version of ``rasa_core_sdk`` during installation

[0.14.1] - 2019-05-02
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed MappingPolicy bug upon prediction of ACTION_LISTEN after mapped action

[0.14.0] - 2019-04-23
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``tf.ConfigProto`` configuration can now be specified
  for tensorflow based pipelines
- open api spec for the Rasa Core SDK action server
- documentation about early deactivation of a form in validation
- Added max_event_history in tracker_store to set this value in DialogueStateTracker
- utility functions for colored logging
- open webbrowser when visualizing stories
- added ``/parse`` endpoint to query for NLU results
- File based event store
- ability to configure event store using the endpoints file
- added ability to use multiple env vars per line in yaml files
- added ``priority`` property of policies to influence best policy in
  the case of equal confidence
- **support for python 3.7**
- ``Tracker.active_form`` now includes ``trigger_message`` attribute to allow
  access to message triggering the form
- ``MappingPolicy`` which can be used to directly map an intent to an action
  by adding the ``triggers`` keyword to an intent in the domain.
- default action ``action_back``, which when triggered with ``/back`` allows
  the user to undo their previous message

Changed
-------
- starter packs are now tested in parallel with the unittests,
  and only on master and branches ending in ``.x`` (i.e. new version releases)
- renamed ``train_dialogue_model`` to ``train``
- renamed ``rasa_core.evaluate`` to ``rasa_core.test``
- ``event_broker.publish`` receives the event as a dict instead of text
- configuration key ``store_type`` of the tracker store endpoint configuration
  has been renamed to ``type`` to allow usage across endpoints
- renamed ``policy_metadata.json`` to ``metadata.json`` for persisted models
- ``scores`` array returned by the ``/conversations/{sender_id}/predict``
  endpoint is now sorted according to the actions' scores.
- now randomly created augmented stories are subsampled during training and marked,
  so that memo policies can ignore them
- changed payloads from "text" to "message" in files: server.yml, docs/connectors.rst,
  rasa_core/server.py, rasa_core/training/interactive.py, tests/test_interactive.py
- dialogue files in ``/data/test_dialogues`` were updated with conversations
  from the bots in ``/examples``
- updated to tensorflow 1.13

Removed
-------
- removed ``admin_token`` from ``RasaChatInput`` since it wasn't used

Fixed
-----
- When a ``fork`` is used in interactive learning, every forked
  storyline is saved (not just the last)
- Handles slot names which contain characters that are invalid as python
  variable name (e.g. dot) in a template

[0.13.8] - 2019-04-16
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Message parse data no longer passed to graph node label in interactive
  learning visualization

[0.13.7] - 2019-04-01
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- correctly process form actions in end-to-end evaluations

[0.13.6] - 2019-03-28
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- correctly process intent messages in end-to-end evaluations

[Unreleased 0.13.8.aX]
^^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Message parse data no longer passed to graph node label in interactive
  learning visualization

[0.13.7] - 2019-04-01
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- correctly process form actions in end-to-end evaluations

[0.13.6] - 2019-03-28
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- correctly process intent messages in end-to-end evaluations

[0.13.4] - 2019-03-19
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- properly tag docker image as ``stable`` (instead of tagging alpha tags)

[0.13.3] - 2019-03-04
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- Tracker Store Mongo DB's documentation now has ``auth_source`` parameter,
  which is used for passing database name associated with the user's
  credentials.

[0.13.2] - 2019-02-06
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- ``MessageProcessor`` now also passes ``message_id`` to the interpreter
  when parsing with a ``RasaNLUHttpInterpreter``

[0.13.1] - 2019-01-29
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``message_id`` can now be passed in the payload to the
  ``RasaNLUHttpInterpreter``

Fixed
-----
- fixed domain persistence after exiting interactive learning
- fix form validation question error in interactive learning

.. _corev0-13-0:

[0.13.0] - 2019-01-23
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- A support for session persistence mechanism in the ``SocketIOInput``
  compatible with the example SocketIO WebChat + short explanation on
  how session persistence should be implemented in a frontend
- ``TwoStageFallbackPolicy`` which asks the user for their affirmation
  if the NLU confidence is low for an intent, for rephrasing the intent
  if they deny the suggested intent, and does finally an ultimate fallback
  if it does not get the intent right
- Additional checks in PolicyEnsemble to ensure that custom Policy
  classes' ``load`` function returns the correct type
- Travis script now clones and tests the Rasa stack starter pack
- Entries for tensorflow and sklearn versions to the policy metadata
- SlackInput wont ignore ``app_mention`` event anymore.
  Will handle messages containing @mentions to bots and will respond to these
  (as long as the event itself is enabled in the application hosting the bot)
- Added sanitization mechanism for SlackInput that (in its current shape and form)
  strips bot's self mentions from messages posted using the said @mentions.
- Added sanitization mechanism for SlackInput that (in its current
  shape and form) strips bot's self mentions from messages posted using
  the said @mentions.
- Added random seed option for KerasPolicy and EmbeddingPolicy
  to allow for reproducible training results
- ``InvalidPolicyConfig`` error if policy in policy configuration could not be
  loaded, or if ``policies`` key is empty or not provided
- Added a unique identifier to ``UserMessage`` and the ``UserUttered`` event.

Removed
-------
- removed support for deprecated intents/entities format

Changed
-------
- replaced ``pytest-pep8`` with ``pytest-pycodestyle``
- switch from ``PyInquirer`` to ``questionary`` for the display of
  commandline interface (to avoid prompt toolkit 2 version issues)
- if NLU classification returned ``None`` in interactive training,
  directly ask a user for a correct intent
- trigger ``fallback`` on low nlu confidence
  only if previous action is ``action_listen``
- updated docs for interactive learning to inform users of the
  ``--core`` flag
- Change memoization policies confidence score to 1.1 to override ML policies
- replaced flask server with async sanic

Fixed
-----
- fix error during interactive learning which was caused by actions which
  dispatched messages using ``dispatcher.utter_custom_message``
- re-added missing ``python-engineio`` dependency
- fixed not working examples in ``examples/``
- strip newlines from messages so you don't have something like "\n/restart\n"
- properly reload domain when using ``/model`` endpoint to upload new model
- updated documentation for custom channels to use the ``credentials.yml``

[0.12.3] - 2018-12-03
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- added ``scipy`` dependency (previously pulled in through keras)
- added element representation for command-line output

Changed
-------
- improved button representation for custom buttons in command-line

Changed
-------
- randomized initial sender_id during interactive training to avoid
  loading previous sessions from persistent tracker stores

Removed
-------
- removed keras dependency, since ``keras_policy`` uses ``tf.keras``


[0.12.2] - 2018-11-20
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- argument handling on evaluate script
- added basic sanitization during visualization


[0.12.1] - 2018-11-11
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed interactive learning to properly submit executed actions to the action
  server
- allow the specification of the policy configuration while using the
  visualisation script
- use default configuration if no policy configuration is passed
- fixed html delivery from interactive server script (package compatible)
- ``SlackBot`` when created in ``SlackInputChannel`` inherits the
  ``slack_channel`` property, allowing Slack bots to post to any channel
  instead of only back to the user
- fix writing of new domain file from interactive learning
- fix reading of state featurizers from yaml
- fix reading of batch_size parameter in keras policy


.. _corev0-12-0:

[0.12.0] - 2018-11-11
^^^^^^^^^^^^^^^^^^^^^

.. warning::

    This is major new version with a lot of changes under the hood as well
    as on the API level. Please take a careful look at the
    :ref:`migration-guide` guide before updating. **You need to retrain your models.**

Added
-----
- new connector for the Cisco Webex Teams chat
- openapi documentation of server API
- NLU data learned through interactive learning will now be stored in a
  separate markdown-format file (any previous NLU data is merged)
- Command line interface for interactive learning now displays policy
  confidence alongside the action name
- added action prediction confidence & policy to ``ActionExecuted`` event
- the Core policy configuration can now be set in a config.yaml file.
  This makes training custom policies possible.
- both the date and the time at which a model was trained are now
  included in the policy's metadata when it is persisted
- show visualization of conversation while doing interactive learning
- option for end-to-end evaluation of Rasa Core and NLU examples in
  ``evaluate.py`` script
- `/conversations/{sender_id}/story` endpoint for returning
  the end-to-end story describing a conversation
- docker-compose file to start a rasa core server together with nlu,
  an action server, and duckling
- http server (``rasa_core.run --enable-api``) evaluation endpoint
- ability to add tracker_store using endpoints.yml
- ability load custom tracker store modules using the endpoints.yml
- ability to add an event broker using an endpoint configuration file
- raise an exception when ``server.py`` is used instead of
  ``rasa_core.run --enable-api``
- add documentation on how to configure endpoints within a configuration file
- ``auth_source`` parameter in ``MongoTrackerStore`` defining the database to
  authenticate against
- missing instructions on setting up the facebook connector
- environment variables specified with ``${env_variable}`` in a yaml
  configuration file are now replaced with the value of the
  environment variable
- detailed documentation on how to deploy Rasa with Docker
- make ``wait_time_between_pulls`` configurable through endpoint
  configuration
- add ``FormPolicy`` to handle form action prediction
- add ``ActionExecutionRejection`` exception and
  ``ActionExecutionRejected`` event
- add default action ``ActionDeactivateForm()``
- add ``formbot`` example
- add ability to turn off auto slot filling with entity for each
  slot in domain.yml
- add ``InvalidDomain`` exception
- add ``active_form_...`` to state dictionary
- add ``active_form`` and ``latest_action_name`` properties to
  ``DialogueStateTracker``
- add ``Form`` and ``FormValidation`` events
- add ``REQUESTED_SLOT`` constant
- add ability to read ``action_listen`` from stories
- added train/eval scripts to compare policies

Changed
-------
- improved response format for ``/predict`` endpoint
- all error messages from the server are now in json format
- ``agent.log_message`` now returns a tracker instead of the trackers state
- the core container does not load the nlu model by default anymore.
  Instead it can be connected to a nlu server.
- stories are now visualized as ``.html`` page instead of an image
- move and deduplicate restaurantbot nlu data from ``franken_data.json``
  to ``nlu_data.md``
- forms were completely reworked, see changelog in ``rasa_core_sdk``
- state featurization if some form is active changed
- ``Domain`` raises ``InvalidDomain`` exception
- interactive learning is now started with rasa_core.train interactive
- passing a policy config file to train a model is now required
- flags for output of evaluate script have been merged to one flag ``--output``
  where you provide a folder where any output from the script should be stored

Removed
-------
- removed graphviz dependency
- policy config related flags in training script (see migration guide)


Fixed
-----
- fixed an issue with boolean slots where False and None had the same value
  (breaking model compatibility with models that use a boolean slot)
- use utf8 everywhere when handling file IO
- argument ``--connector`` on run script accepts custom channel module names
- properly handle non ascii categorical slot values, e.g. ``大于100亿元``
- fixed HTTP server attempting to authenticate based on incorrect path to
  the correct JWT data field
- all sender ids from channels are now handled as `str`.
  Sender ids from old messages with an `int` id are converted to `str`.
- legacy pep8 errors


[0.11.12] - 2018-10-11
^^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- Remove livechat widget from docs


[0.11.11] - 2018-10-05
^^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Add missing name() to facebook Messenger class


[0.11.10] - 2018-10-05
^^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- backport fix to JWT schema


[0.11.9] - 2018-10-04
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- pin tensorflow 1.10.0

[0.11.8] - 2018-09-28
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- cancel reminders if there has been a restarted event after the reminder

Changed
-------
- JWT authentication now checks user roles. The ``admin`` role may access all
  endpoints. For endpoints which contain a ``sender_id`` parameter, users
  with the ``user`` role may only call endpoints where the ``sender_id``
  matches the user's ``username``.

[0.11.7] - 2018-09-26
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- custom message method in rocketchat channel

Fixed
-----
- don't fail if rasa and rest input channels are used together
- wrong paramter name in rocketchat channel methods
- Software 2.0 link on interactive learning documentation page went to
  Tesla's homepage, now it links to Karpathy blogpost

[0.11.6] - 2018-09-20
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``UserMessage`` and ``UserUttered`` classes have a new attribute
  ``input_channel`` that stores the name of the ``InputChannel``
  through which the message was received

[0.11.5] - 2018-09-20
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- numpy version incompatibility between rasa core and tensorflow

[0.11.4] - 2018-09-19
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- a flag ``--fail_on_prediction_errors`` to the ``evaluate.py`` script -
  if used when running the evaluation, the script will fail with a non
  0 exit code if there is at least one prediction error. This can be
  used on CIs to validate models against test stories.
- JWT support: parameters to allow clients to authenticate requests to
  the rasa_core.server using JWT's in addition to normal token based auth
- added socket.io input / output channel
- ``UserMessage`` and ``UserUttered`` classes have a new attribute
  ``input_channel`` that stores the name of the ``InputChannel``
  through which the message was received

Changed
-------
- dump failed stories after evaluation in the normal story format instead of
  as a text file
- do not run actions during evaluation. instead, action are only predicted
  and validated against the gold story.
- improved the online learning experience on the CLI
- made finetuning during online learning optional (use ``--finetune`` if
  you want to enable it)

Removed
-------
- package pytest-services since it wasn't necessary

Fixed
-----
- fixed an issue with the followup (there was a name confusion, sometimes
  the followup action would be set to the non existent ``follow_up_action``
  attribute instead of ``followup_action``)

[0.11.3] - 2018-09-04
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- callback output channel, receives messages and uses a REST endpoint to
  respond with messages

Changed
-------
- channel input creation moved to the channel, every channel can now
  customize how it gets created from the credentials file

[0.11.2] - 2018-09-04
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- improved documentation for events (e.g. including json serialisation)

Removed
-------
- outdated documentation for removed endpoints in the server
  (``/parse`` & ``/continue``)

Fixed
-----
- read in fallback command line args

[0.11.1] - 2018-08-30
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- increased minimal compatible model version to 0.11.0

.. _corev0-11-0:

[0.11.0] - 2018-08-30
^^^^^^^^^^^^^^^^^^^^^

.. warning::

    This is major new version with a lot of changes under the hood as well
    as on the API level. Please take a careful look at the
    :ref:`migration-guide` guide before updating. You need to retrain your models.


Added
-----
- added microsoft botframework input and output channels
- added rocket chat input and output channels
- script parameter ``--quiet`` to set the log level to ``WARNING``
- information about the python version a model has been trained with to the
  model metadata
- more emoji support for PY2
- intent confidence support in RegexInterpreter
- added paramter to train script to pull training data from an url instead
  of a stories file
- added new policy: :ref:`embedding_policy` implemented in tensorflow

Changed
-------
- default log level for all scripts has been changed from ``WARNING`` to
  ``INFO``.
- format of the credentials file to allow specifying the credentials for
  multiple channels
- webhook URLs for the input channels have changed and need to be reset
- deprecated using ``rasa_core.server`` as a script - use
  ``rasa_core.run --enable_api`` instead
- collecting output channel will no properly collect events for images,
  buttons, and attachments

Removed
-------
- removed the deprecated ``TopicSet`` event
- removed ``tracker.follow_up_action`` - use the ``FollowupAction``
  event instead
- removed ``action_factory: remote`` from domain file - the domain is
  always run over http
- removed ``OnlineLearningPolicy`` - use the ``training.online``
  script instead

Fixed
-------
- lots of type annotations
- some invalid documentation references
- changed all ``logger.warn`` to ``logger.warning``

[0.10.4] - 2018-08-08
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- more emoji support for PY2
- intent confidence support in RegexInterpreter

[0.10.3] - 2018-08-03
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- updated to Rasa NLU 0.13
- improved documentation quickstart

Fixed
-----
- server request argument handling on python 3
- creation of training data story graph - removes more nodes and speeds up
  the training

[0.10.2] - 2018-07-24
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- new ``RasaChatInput`` channel
- option to ignore entities for certain intents

Fixed
-----
- loading of NLU model

[0.10.1] - 2018-07-18
^^^^^^^^^^^^^^^^^^^^^

Changed
-------

- documentation changes

.. _corev0-10-0:

[0.10.0] - 2018-07-17
^^^^^^^^^^^^^^^^^^^^^

.. warning::

    This is a major new release with backward incompatible changes. Old trained
    models can not be read with the new version - you need to retrain your model.
    View the :ref:`migration-guide` for details.

Added
-----
- allow bot responses to be managed externally (instead of putting them into
  the ``domain.yml``)
- options to prevent slack from making re-deliver message upon meeting failure condition.
  the default is to ignore ``http_timeout``.
- added ability to create domain from yaml string and export a domain to a yaml string
- added server endpoint to fetch domain as json or yaml
- new default action ActionDefaultFallback
- event streaming to a ``RabbitMQ`` message broker using ``Pika``
- docs section on event brokers
- ``Agent()`` class supports a ``model_server`` ``EndpointConfig``, which it regularly queries to fetch dialogue models
- this can be used with ``rasa_core.server`` with the ``--endpoint`` option (the key for this the model server config is ``model``)
- docs on model fetching from a URL

Changed
-------
- changed the logic inside AugmentedMemoizationPolicy to recall actions only if they are the same in training stories
- moved AugmentedMemoizationPolicy to memoization.py
- wrapped initialization of BackgroundScheduler in try/except to allow running on jupyterhub / binderhub/ colaboratory
- fixed order of events logged on a tracker: action executed is now always
  logged before bot utterances that action created

Removed
-------
- removed support for topics

[0.9.6] - 2018-06-18
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed fallback policy data generation

[0.9.5] - 2018-06-14
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- handling of max history configuration in policies
- fixed instantiation issues of fallback policy

[0.9.4] - 2018-06-07
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed evaluation script
- fixed story file loading (previously some story files with checkpoints could
  create wrong training data)
- improved speed of data loading

[0.9.3] - 2018-05-30
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- added token auth to all endpoints of the core server


[0.9.2] - 2018-05-30
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fix handling of max_history parameter in AugmentedMemoizationPolicy

[0.9.1] - 2018-05-29
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- persistence of training data collected during online learning if default
  file path is used
- the ``agent()`` method used in some ``rasa_core.server`` endpoints is
  re-run at every new call of the ``ensure_loaded_agent`` decorator
- fixed OR usage of intents

.. _corev0-9-0:

[0.9.0] - 2018-05-24
^^^^^^^^^^^^^^^^^^^^

.. warning::

    This is a major new release with backward incompatible changes. Old trained
    models can not be read with the new version - you need to retrain your model.

Added
-----
- supported loading training data from a folder - loads all stories from
  all files in that directory
- parameter to specify NLU project when instantiating a ``RasaNLUInterpreter``
- simple ``/respond`` endpoint to get bot response to a user message
- ``/conversations`` endpoint for listing sender ids of running conversations
- added a Mattermost channel that allows Rasa Core to communicate via a Mattermost app
- added a Twilio channel that allows Rasa Core to communicate via SMS
- ``FallbackPolicy`` for executing a default message if NLU or core model confidence is low.
- ``FormAction`` class to make it easier to collect multiple pieces of information with fewer stories.
- Dockerfile for ``rasa_core.server`` with a dialogue and Rasa NLU model

Changed
-------
- moved server from klein to flask
- updated dependency fbmessenger from 4.3.1 to 5.0.0
- updated Rasa NLU to 0.12.x
- updated all the dependencies to the latest versions

Fixed
-----
- List slot is now populated with a list
- Slack connector: ``slack_channel`` kwarg is used to send messages either back to the user or to a static channel
- properly log to a file when using the ``run`` script
- documentation fix on stories


[0.8.6] - 2018-04-18
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- pin rasa nlu version to 0.11.4 (0.12.x only works with master)

[0.8.5] - 2018-03-19
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- updated google analytics docs survey code


[0.8.4] - 2018-03-14
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- pin ``pykwalify<=1.6.0`` as update to ``1.6.1`` breaks compatibility

[0.8.3] - 2018-02-28
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- pin ``fbmessenger`` version to avoid major update

[0.8.2] - 2018-02-13
^^^^^^^^^^^^^^^^^^^^

Added
-----
- script to reload a dumped trackers state and to continue the conversation
  at the end of the stored dialogue

Changed
-------
- minor updates to dependencies

Fixed
-----
- fixed datetime serialisation of reminder event

[0.8.1] - 2018-02-01
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- removed deque to support python 3.5
- Documentation improvements to tutorials
- serialisation of date time value for ``ReminderScheduled`` event

.. _corev0-8-0:

[0.8.0] - 2018-01-30
^^^^^^^^^^^^^^^^^^^^

This is a major version change. Make sure to take a look at the
:ref:`migration-guide` in the documentation for advice on how to
update existing projects.

Added
-----
- ``--debug`` and ``--verbose`` flags to scripts (train.py, run.py, server.py)
  to set the log level
- support for story cycles when using checkpoints
- added a new machine learning policy `SklearnPolicy` that uses an sklearn
  classifier to predict actions (logistic regression by default)
- warn if action emits events when using a model that it did never emit in
  any of the stories the model was trained on
- support for event pushing and endpoints to retrieve the tracker state from the server
- Timestamp to every event
- added a Slack channel that allows Rasa Core to communicate via a Slack app
- added a Telegram channel that allows Rasa Core to communicate via a Telegram bot

Changed
-------
- rewrite of the whole FB connector: replaced pymessenger library with fbmessenger
- story file utterance format changed from ``* _intent_greet[name=Rasa]``
  to ``* intent_greet{"name": "Rasa"}`` (old format is still supported but
  deprecated)
- persist action names in domain during model persistence
- improved travis build speed by not using miniconda
- don't fail with an exception but with a helpful error message if an
  utterance template contains a variable that can not be filled
- domain doesn't fail on unknown actions but emits a warning instead. this is to support reading
  logs from older conversation if one recently removed an action from the domain

Fixed
-----
- proper evaluation of stories with checkpoints
- proper visualisation of stories with checkpoints
- fixed float slot min max value handling
- fixed non integer feature decoding, e.g. used for memoization policy
- properly log to specified file when starting Rasa Core server
- properly calculate offset of last reset event after loading tracker from
  tracker store
- UserUtteranceReverted action incorrectly triggered actions to be replayed


[0.7.9] - 2017-11-29
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- visualisation using Networkx version 2.x
- add output about line of failing intent when parsing story files

[0.7.8] - 2017-11-27
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Pypi readme rendering

[0.7.7] - 2017-11-24
^^^^^^^^^^^^^^^^^^^^

Added
-----
- log bot utterances to tracker

Fixed
-----
- documentation improvements in README
- renamed interpreter argument to rasa core server

[0.7.6] - 2017-11-15
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- moodbot example train command in docs


[0.7.5] - 2017-11-14
^^^^^^^^^^^^^^^^^^^^

Changed
-------
- "sender_id" (and "DEFAULT_SENDER_ID") keyword consistency issue #56

Fixed
-----
- improved moodbot example - more nlu examples as well as better fitting of dialogue model


[0.7.4] - 2017-11-09
^^^^^^^^^^^^^^^^^^^^

Changed
-------

- added method to tracker to retrieve the latest entities #68

[0.7.3] - 2017-10-31
^^^^^^^^^^^^^^^^^^^^

Added
-----
- parameter to specify font size when rendering story visualization

Fixed
-----
- fixed documentation of story visualization

[0.7.2] - 2017-10-30
^^^^^^^^^^^^^^^^^^^^

Added
-----
- added facebook bot example
- added support for conditional checkpoints. a checkpoint can be restricted to
  only allow one to use it if certain slots are set. see docs for details
- utterance templates in domain yaml support buttons and images
- validate domain yaml and raise exception on invalid file
- ``run`` script to load models and handle messages from an input channel

Changed
-------
- small dropout in standard keras model to decrease reliance on exact intents
- a LOT of documentation improvements

Fixed
-----
- fixed http error if action listen is not confirmed. #42

[0.7.1] - 2017-10-06
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- issues with restart events. They created wrong a messed up history leading to
  wrong predictions


.. _corev0-7-0:

[0.7.0] - 2017-10-04
^^^^^^^^^^^^^^^^^^^^

Added
-----
- support for Rasa Core usage as a server with remote action execution

Changed
-------
- switched to max code line length 80
- removed action id - use ``action.name()`` instead. if an action implementation overrides the name, it should include the ``action_`` prefix (as it is not automatically added anymore)
- renamed ``rasa_dm.util`` to ``rasa_dm.utils``
- renamed the whole package to ``rasa_core`` (so ``rasa_dm`` is gone!)
- renamed ``Reminder`` attribute ``id`` to ``name``
- a lot of documentation improvements. docs are now at https://rasa.com/docs/core
- use hashing when writing memorized turns into persistence - requires retraining of all models that are trained with a version prior to this
- changed ``agent.handle_message(...)`` interface for easier usage

.. _corev0-6-0:

[0.6.0] - 2017-08-27
^^^^^^^^^^^^^^^^^^^^

Added
-----
- support for multiple policies (e.g. one memoization and a Keras policy at the same time)
- loading domains from yaml files instead of defining them with python code
- added an api layer (called ``Agent``) for you to use for 95% of the things you want to do (training, persistence, loading models)
- support for reminders

Changed
-------
- large refactoring of code base

.. _corev0-5-0:

[0.5.0] - 2017-06-18
^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``ScoringPolicy`` added to policy implementations (less strict than standard default policy)
- ``RasaNLUInterpreter`` to run a nlu instance within dm (instead of using the http interface)
- more tests

Changed
-------
- ``UserUtterance`` now holds the complete parse data from nlu (e.g. to access attributes other than entities or intent)
- ``Turn`` has a reference to a ``UserUtterance`` instead of directly storing intent & entities (allows access to other data)
- Simplified interface of output channels
- order of actions in the DefaultPolicy in ``possible_actions`` (``ActionListen`` now always has index 0)

Fixed
-----
- ``RedisTrackerStore`` checks if tracker is stored before accessing it (otherwise a ``None`` access exception is thrown)
- ``RegexInterpreter`` checks if the regex actually matches the message instead of assuming it always does
- ``str`` implementation for all events
- ``Controller`` can be started without an input channel (e.g. messages need to be fed into the queue manually)

.. _corev0-2-0:

[0.2.0] - 2017-05-18
^^^^^^^^^^^^^^^^^^^^
First released version.


.. _`master`: https://github.com/RasaHQ/rasa_core/

.. _`Semantic Versioning`: http://semver.org/
