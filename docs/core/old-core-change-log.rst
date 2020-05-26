:desc: Rasa Core Changelog a
 a
.. _old-core-change-log: a
 a
Core Change Log a
=============== a
 a
All notable changes to this project will be documented in this file. a
This project adheres to `Semantic Versioning`_ starting with version 0.2.0. a
 a
[0.14.4] - 2019-05-13 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- correctly process form actions in core evaluations a
 a
[0.14.3] - 2019-05-07 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed interactive learning history printing a
 a
[0.14.2] - 2019-05-07 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed required version of ``rasa_core_sdk`` during installation a
 a
[0.14.1] - 2019-05-02 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed MappingPolicy bug upon prediction of ACTION_LISTEN after mapped action a
 a
[0.14.0] - 2019-04-23 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- ``tf.ConfigProto`` configuration can now be specified a
  for tensorflow based pipelines a
- open api spec for the Rasa Core SDK action server a
- documentation about early deactivation of a form in validation a
- Added max_event_history in tracker_store to set this value in DialogueStateTracker a
- utility functions for colored logging a
- open webbrowser when visualizing stories a
- added ``/parse`` endpoint to query for NLU results a
- File based event store a
- ability to configure event store using the endpoints file a
- added ability to use multiple env vars per line in yaml files a
- added ``priority`` property of policies to influence best policy in a
  the case of equal confidence a
- **support for python 3.7** a
- ``Tracker.active_form`` now includes ``trigger_message`` attribute to allow a
  access to message triggering the form a
- ``MappingPolicy`` which can be used to directly map an intent to an action a
  by adding the ``triggers`` keyword to an intent in the domain. a
- default action ``action_back``, which when triggered with ``/back`` allows a
  the user to undo their previous message a
 a
Changed a
------- a
- starter packs are now tested in parallel with the unittests, a
  and only on master and branches ending in ``.x`` (i.e. new version releases) a
- renamed ``train_dialogue_model`` to ``train`` a
- renamed ``rasa_core.evaluate`` to ``rasa_core.test`` a
- ``event_broker.publish`` receives the event as a dict instead of text a
- configuration key ``store_type`` of the tracker store endpoint configuration a
  has been renamed to ``type`` to allow usage across endpoints a
- renamed ``policy_metadata.json`` to ``metadata.json`` for persisted models a
- ``scores`` array returned by the ``/conversations/{sender_id}/predict`` a
  endpoint is now sorted according to the actions' scores. a
- now randomly created augmented stories are subsampled during training and marked, a
  so that memo policies can ignore them a
- changed payloads from "text" to "message" in files: server.yml, docs/connectors.rst, a
  rasa_core/server.py, rasa_core/training/interactive.py, tests/test_interactive.py a
- dialogue files in ``/data/test_dialogues`` were updated with conversations a
  from the bots in ``/examples`` a
- updated to tensorflow 1.13 a
 a
Removed a
------- a
- removed ``admin_token`` from ``RasaChatInput`` since it wasn't used a
 a
Fixed a
----- a
- When a ``fork`` is used in interactive learning, every forked a
  storyline is saved (not just the last) a
- Handles slot names which contain characters that are invalid as python a
  variable name (e.g. dot) in a template a
 a
[0.13.8] - 2019-04-16 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Message parse data no longer passed to graph node label in interactive a
  learning visualization a
 a
[0.13.7] - 2019-04-01 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- correctly process form actions in end-to-end evaluations a
 a
[0.13.6] - 2019-03-28 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- correctly process intent messages in end-to-end evaluations a
 a
[Unreleased 0.13.8.aX] a
^^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Message parse data no longer passed to graph node label in interactive a
  learning visualization a
 a
[0.13.7] - 2019-04-01 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- correctly process form actions in end-to-end evaluations a
 a
[0.13.6] - 2019-03-28 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- correctly process intent messages in end-to-end evaluations a
 a
[0.13.4] - 2019-03-19 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- properly tag docker image as ``stable`` (instead of tagging alpha tags) a
 a
[0.13.3] - 2019-03-04 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- Tracker Store Mongo DB's documentation now has ``auth_source`` parameter, a
  which is used for passing database name associated with the user's a
  credentials. a
 a
[0.13.2] - 2019-02-06 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- ``MessageProcessor`` now also passes ``message_id`` to the interpreter a
  when parsing with a ``RasaNLUHttpInterpreter`` a
 a
[0.13.1] - 2019-01-29 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- ``message_id`` can now be passed in the payload to the a
  ``RasaNLUHttpInterpreter`` a
 a
Fixed a
----- a
- fixed domain persistence after exiting interactive learning a
- fix form validation question error in interactive learning a
 a
.. _corev0-13-0: a
 a
[0.13.0] - 2019-01-23 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- A support for session persistence mechanism in the ``SocketIOInput`` a
  compatible with the example SocketIO WebChat + short explanation on a
  how session persistence should be implemented in a frontend a
- ``TwoStageFallbackPolicy`` which asks the user for their affirmation a
  if the NLU confidence is low for an intent, for rephrasing the intent a
  if they deny the suggested intent, and does finally an ultimate fallback a
  if it does not get the intent right a
- Additional checks in PolicyEnsemble to ensure that custom Policy a
  classes' ``load`` function returns the correct type a
- Travis script now clones and tests the Rasa stack starter pack a
- Entries for tensorflow and sklearn versions to the policy metadata a
- SlackInput wont ignore ``app_mention`` event anymore. a
  Will handle messages containing @mentions to bots and will respond to these a
  (as long as the event itself is enabled in the application hosting the bot) a
- Added sanitization mechanism for SlackInput that (in its current shape and form) a
  strips bot's self mentions from messages posted using the said @mentions. a
- Added sanitization mechanism for SlackInput that (in its current a
  shape and form) strips bot's self mentions from messages posted using a
  the said @mentions. a
- Added random seed option for KerasPolicy and EmbeddingPolicy a
  to allow for reproducible training results a
- ``InvalidPolicyConfig`` error if policy in policy configuration could not be a
  loaded, or if ``policies`` key is empty or not provided a
- Added a unique identifier to ``UserMessage`` and the ``UserUttered`` event. a
 a
Removed a
------- a
- removed support for deprecated intents/entities format a
 a
Changed a
------- a
- replaced ``pytest-pep8`` with ``pytest-pycodestyle`` a
- switch from ``PyInquirer`` to ``questionary`` for the display of a
  commandline interface (to avoid prompt toolkit 2 version issues) a
- if NLU classification returned ``None`` in interactive training, a
  directly ask a user for a correct intent a
- trigger ``fallback`` on low nlu confidence a
  only if previous action is ``action_listen`` a
- updated docs for interactive learning to inform users of the a
  ``--core`` flag a
- Change memoization policies confidence score to 1.1 to override ML policies a
- replaced flask server with async sanic a
 a
Fixed a
----- a
- fix error during interactive learning which was caused by actions which a
  dispatched messages using ``dispatcher.utter_custom_message`` a
- re-added missing ``python-engineio`` dependency a
- fixed not working examples in ``examples/`` a
- strip newlines from messages so you don't have something like "\n/restart\n" a
- properly reload domain when using ``/model`` endpoint to upload new model a
- updated documentation for custom channels to use the ``credentials.yml`` a
 a
[0.12.3] - 2018-12-03 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- added ``scipy`` dependency (previously pulled in through keras) a
- added element representation for command-line output a
 a
Changed a
------- a
- improved button representation for custom buttons in command-line a
 a
Changed a
------- a
- randomized initial sender_id during interactive training to avoid a
  loading previous sessions from persistent tracker stores a
 a
Removed a
------- a
- removed keras dependency, since ``keras_policy`` uses ``tf.keras`` a
 a
 a
[0.12.2] - 2018-11-20 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- argument handling on evaluate script a
- added basic sanitization during visualization a
 a
 a
[0.12.1] - 2018-11-11 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed interactive learning to properly submit executed actions to the action a
  server a
- allow the specification of the policy configuration while using the a
  visualisation script a
- use default configuration if no policy configuration is passed a
- fixed html delivery from interactive server script (package compatible) a
- ``SlackBot`` when created in ``SlackInputChannel`` inherits the a
  ``slack_channel`` property, allowing Slack bots to post to any channel a
  instead of only back to the user a
- fix writing of new domain file from interactive learning a
- fix reading of state featurizers from yaml a
- fix reading of batch_size parameter in keras policy a
 a
 a
.. _corev0-12-0: a
 a
[0.12.0] - 2018-11-11 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
.. warning:: a
 a
    This is major new version with a lot of changes under the hood as well a
    as on the API level. Please take a careful look at the a
    :ref:`migration-guide` guide before updating. **You need to retrain your models.** a
 a
Added a
----- a
- new connector for the Cisco Webex Teams chat a
- openapi documentation of server API a
- NLU data learned through interactive learning will now be stored in a a
  separate markdown-format file (any previous NLU data is merged) a
- Command line interface for interactive learning now displays policy a
  confidence alongside the action name a
- added action prediction confidence & policy to ``ActionExecuted`` event a
- the Core policy configuration can now be set in a config.yaml file. a
  This makes training custom policies possible. a
- both the date and the time at which a model was trained are now a
  included in the policy's metadata when it is persisted a
- show visualization of conversation while doing interactive learning a
- option for end-to-end evaluation of Rasa Core and NLU examples in a
  ``evaluate.py`` script a
- `/conversations/{sender_id}/story` endpoint for returning a
  the end-to-end story describing a conversation a
- docker-compose file to start a rasa core server together with nlu, a
  an action server, and duckling a
- http server (``rasa_core.run --enable-api``) evaluation endpoint a
- ability to add tracker_store using endpoints.yml a
- ability load custom tracker store modules using the endpoints.yml a
- ability to add an event broker using an endpoint configuration file a
- raise an exception when ``server.py`` is used instead of a
  ``rasa_core.run --enable-api`` a
- add documentation on how to configure endpoints within a configuration file a
- ``auth_source`` parameter in ``MongoTrackerStore`` defining the database to a
  authenticate against a
- missing instructions on setting up the facebook connector a
- environment variables specified with ``${env_variable}`` in a yaml a
  configuration file are now replaced with the value of the a
  environment variable a
- detailed documentation on how to deploy Rasa with Docker a
- make ``wait_time_between_pulls`` configurable through endpoint a
  configuration a
- add ``FormPolicy`` to handle form action prediction a
- add ``ActionExecutionRejection`` exception and a
  ``ActionExecutionRejected`` event a
- add default action ``ActionDeactivateForm()`` a
- add ``formbot`` example a
- add ability to turn off auto slot filling with entity for each a
  slot in domain.yml a
- add ``InvalidDomain`` exception a
- add ``active_form_...`` to state dictionary a
- add ``active_form`` and ``latest_action_name`` properties to a
  ``DialogueStateTracker`` a
- add ``Form`` and ``FormValidation`` events a
- add ``REQUESTED_SLOT`` constant a
- add ability to read ``action_listen`` from stories a
- added train/eval scripts to compare policies a
 a
Changed a
------- a
- improved response format for ``/predict`` endpoint a
- all error messages from the server are now in json format a
- ``agent.log_message`` now returns a tracker instead of the trackers state a
- the core container does not load the nlu model by default anymore. a
  Instead it can be connected to a nlu server. a
- stories are now visualized as ``.html`` page instead of an image a
- move and deduplicate restaurantbot nlu data from ``franken_data.json`` a
  to ``nlu_data.md`` a
- forms were completely reworked, see changelog in ``rasa_core_sdk`` a
- state featurization if some form is active changed a
- ``Domain`` raises ``InvalidDomain`` exception a
- interactive learning is now started with rasa_core.train interactive a
- passing a policy config file to train a model is now required a
- flags for output of evaluate script have been merged to one flag ``--output`` a
  where you provide a folder where any output from the script should be stored a
 a
Removed a
------- a
- removed graphviz dependency a
- policy config related flags in training script (see migration guide) a
 a
 a
Fixed a
----- a
- fixed an issue with boolean slots where False and None had the same value a
  (breaking model compatibility with models that use a boolean slot) a
- use utf8 everywhere when handling file IO a
- argument ``--connector`` on run script accepts custom channel module names a
- properly handle non ascii categorical slot values, e.g. ``大于100亿元`` a
- fixed HTTP server attempting to authenticate based on incorrect path to a
  the correct JWT data field a
- all sender ids from channels are now handled as `str`. a
  Sender ids from old messages with an `int` id are converted to `str`. a
- legacy pep8 errors a
 a
 a
[0.11.12] - 2018-10-11 a
^^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- Remove livechat widget from docs a
 a
 a
[0.11.11] - 2018-10-05 a
^^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Add missing name() to facebook Messenger class a
 a
 a
[0.11.10] - 2018-10-05 a
^^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- backport fix to JWT schema a
 a
 a
[0.11.9] - 2018-10-04 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- pin tensorflow 1.10.0 a
 a
[0.11.8] - 2018-09-28 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- cancel reminders if there has been a restarted event after the reminder a
 a
Changed a
------- a
- JWT authentication now checks user roles. The ``admin`` role may access all a
  endpoints. For endpoints which contain a ``sender_id`` parameter, users a
  with the ``user`` role may only call endpoints where the ``sender_id`` a
  matches the user's ``username``. a
 a
[0.11.7] - 2018-09-26 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- custom message method in rocketchat channel a
 a
Fixed a
----- a
- don't fail if rasa and rest input channels are used together a
- wrong paramter name in rocketchat channel methods a
- Software 2.0 link on interactive learning documentation page went to a
  Tesla's homepage, now it links to Karpathy blogpost a
 a
[0.11.6] - 2018-09-20 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- ``UserMessage`` and ``UserUttered`` classes have a new attribute a
  ``input_channel`` that stores the name of the ``InputChannel`` a
  through which the message was received a
 a
[0.11.5] - 2018-09-20 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- numpy version incompatibility between rasa core and tensorflow a
 a
[0.11.4] - 2018-09-19 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- a flag ``--fail_on_prediction_errors`` to the ``evaluate.py`` script - a
  if used when running the evaluation, the script will fail with a non a
  0 exit code if there is at least one prediction error. This can be a
  used on CIs to validate models against test stories. a
- JWT support: parameters to allow clients to authenticate requests to a
  the rasa_core.server using JWT's in addition to normal token based auth a
- added socket.io input / output channel a
- ``UserMessage`` and ``UserUttered`` classes have a new attribute a
  ``input_channel`` that stores the name of the ``InputChannel`` a
  through which the message was received a
 a
Changed a
------- a
- dump failed stories after evaluation in the normal story format instead of a
  as a text file a
- do not run actions during evaluation. instead, action are only predicted a
  and validated against the gold story. a
- improved the online learning experience on the CLI a
- made finetuning during online learning optional (use ``--finetune`` if a
  you want to enable it) a
 a
Removed a
------- a
- package pytest-services since it wasn't necessary a
 a
Fixed a
----- a
- fixed an issue with the followup (there was a name confusion, sometimes a
  the followup action would be set to the non existent ``follow_up_action`` a
  attribute instead of ``followup_action``) a
 a
[0.11.3] - 2018-09-04 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- callback output channel, receives messages and uses a REST endpoint to a
  respond with messages a
 a
Changed a
------- a
- channel input creation moved to the channel, every channel can now a
  customize how it gets created from the credentials file a
 a
[0.11.2] - 2018-09-04 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- improved documentation for events (e.g. including json serialisation) a
 a
Removed a
------- a
- outdated documentation for removed endpoints in the server a
  (``/parse`` & ``/continue``) a
 a
Fixed a
----- a
- read in fallback command line args a
 a
[0.11.1] - 2018-08-30 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- increased minimal compatible model version to 0.11.0 a
 a
.. _corev0-11-0: a
 a
[0.11.0] - 2018-08-30 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
.. warning:: a
 a
    This is major new version with a lot of changes under the hood as well a
    as on the API level. Please take a careful look at the a
    :ref:`migration-guide` guide before updating. You need to retrain your models. a
 a
 a
Added a
----- a
- added microsoft botframework input and output channels a
- added rocket chat input and output channels a
- script parameter ``--quiet`` to set the log level to ``WARNING`` a
- information about the python version a model has been trained with to the a
  model metadata a
- more emoji support for PY2 a
- intent confidence support in RegexInterpreter a
- added paramter to train script to pull training data from an url instead a
  of a stories file a
- added new policy: :ref:`embedding_policy` implemented in tensorflow a
 a
Changed a
------- a
- default log level for all scripts has been changed from ``WARNING`` to a
  ``INFO``. a
- format of the credentials file to allow specifying the credentials for a
  multiple channels a
- webhook URLs for the input channels have changed and need to be reset a
- deprecated using ``rasa_core.server`` as a script - use a
  ``rasa_core.run --enable_api`` instead a
- collecting output channel will no properly collect events for images, a
  buttons, and attachments a
 a
Removed a
------- a
- removed the deprecated ``TopicSet`` event a
- removed ``tracker.follow_up_action`` - use the ``FollowupAction`` a
  event instead a
- removed ``action_factory: remote`` from domain file - the domain is a
  always run over http a
- removed ``OnlineLearningPolicy`` - use the ``training.online`` a
  script instead a
 a
Fixed a
------- a
- lots of type annotations a
- some invalid documentation references a
- changed all ``logger.warn`` to ``logger.warning`` a
 a
[0.10.4] - 2018-08-08 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- more emoji support for PY2 a
- intent confidence support in RegexInterpreter a
 a
[0.10.3] - 2018-08-03 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- updated to Rasa NLU 0.13 a
- improved documentation quickstart a
 a
Fixed a
----- a
- server request argument handling on python 3 a
- creation of training data story graph - removes more nodes and speeds up a
  the training a
 a
[0.10.2] - 2018-07-24 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- new ``RasaChatInput`` channel a
- option to ignore entities for certain intents a
 a
Fixed a
----- a
- loading of NLU model a
 a
[0.10.1] - 2018-07-18 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
 a
- documentation changes a
 a
.. _corev0-10-0: a
 a
[0.10.0] - 2018-07-17 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
.. warning:: a
 a
    This is a major new release with backward incompatible changes. Old trained a
    models can not be read with the new version - you need to retrain your model. a
    View the :ref:`migration-guide` for details. a
 a
Added a
----- a
- allow bot responses to be managed externally (instead of putting them into a
  the ``domain.yml``) a
- options to prevent slack from making re-deliver message upon meeting failure condition. a
  the default is to ignore ``http_timeout``. a
- added ability to create domain from yaml string and export a domain to a yaml string a
- added server endpoint to fetch domain as json or yaml a
- new default action ActionDefaultFallback a
- event streaming to a ``RabbitMQ`` message broker using ``Pika`` a
- docs section on event brokers a
- ``Agent()`` class supports a ``model_server`` ``EndpointConfig``, which it regularly queries to fetch dialogue models a
- this can be used with ``rasa_core.server`` with the ``--endpoint`` option (the key for this the model server config is ``model``) a
- docs on model fetching from a URL a
 a
Changed a
------- a
- changed the logic inside AugmentedMemoizationPolicy to recall actions only if they are the same in training stories a
- moved AugmentedMemoizationPolicy to memoization.py a
- wrapped initialization of BackgroundScheduler in try/except to allow running on jupyterhub / binderhub/ colaboratory a
- fixed order of events logged on a tracker: action executed is now always a
  logged before bot utterances that action created a
 a
Removed a
------- a
- removed support for topics a
 a
[0.9.6] - 2018-06-18 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed fallback policy data generation a
 a
[0.9.5] - 2018-06-14 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- handling of max history configuration in policies a
- fixed instantiation issues of fallback policy a
 a
[0.9.4] - 2018-06-07 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed evaluation script a
- fixed story file loading (previously some story files with checkpoints could a
  create wrong training data) a
- improved speed of data loading a
 a
[0.9.3] - 2018-05-30 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- added token auth to all endpoints of the core server a
 a
 a
[0.9.2] - 2018-05-30 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fix handling of max_history parameter in AugmentedMemoizationPolicy a
 a
[0.9.1] - 2018-05-29 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- persistence of training data collected during online learning if default a
  file path is used a
- the ``agent()`` method used in some ``rasa_core.server`` endpoints is a
  re-run at every new call of the ``ensure_loaded_agent`` decorator a
- fixed OR usage of intents a
 a
.. _corev0-9-0: a
 a
[0.9.0] - 2018-05-24 a
^^^^^^^^^^^^^^^^^^^^ a
 a
.. warning:: a
 a
    This is a major new release with backward incompatible changes. Old trained a
    models can not be read with the new version - you need to retrain your model. a
 a
Added a
----- a
- supported loading training data from a folder - loads all stories from a
  all files in that directory a
- parameter to specify NLU project when instantiating a ``RasaNLUInterpreter`` a
- simple ``/respond`` endpoint to get bot response to a user message a
- ``/conversations`` endpoint for listing sender ids of running conversations a
- added a Mattermost channel that allows Rasa Core to communicate via a Mattermost app a
- added a Twilio channel that allows Rasa Core to communicate via SMS a
- ``FallbackPolicy`` for executing a default message if NLU or core model confidence is low. a
- ``FormAction`` class to make it easier to collect multiple pieces of information with fewer stories. a
- Dockerfile for ``rasa_core.server`` with a dialogue and Rasa NLU model a
 a
Changed a
------- a
- moved server from klein to flask a
- updated dependency fbmessenger from 4.3.1 to 5.0.0 a
- updated Rasa NLU to 0.12.x a
- updated all the dependencies to the latest versions a
 a
Fixed a
----- a
- List slot is now populated with a list a
- Slack connector: ``slack_channel`` kwarg is used to send messages either back to the user or to a static channel a
- properly log to a file when using the ``run`` script a
- documentation fix on stories a
 a
 a
[0.8.6] - 2018-04-18 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- pin rasa nlu version to 0.11.4 (0.12.x only works with master) a
 a
[0.8.5] - 2018-03-19 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- updated google analytics docs survey code a
 a
 a
[0.8.4] - 2018-03-14 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- pin ``pykwalify<=1.6.0`` as update to ``1.6.1`` breaks compatibility a
 a
[0.8.3] - 2018-02-28 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- pin ``fbmessenger`` version to avoid major update a
 a
[0.8.2] - 2018-02-13 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- script to reload a dumped trackers state and to continue the conversation a
  at the end of the stored dialogue a
 a
Changed a
------- a
- minor updates to dependencies a
 a
Fixed a
----- a
- fixed datetime serialisation of reminder event a
 a
[0.8.1] - 2018-02-01 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- removed deque to support python 3.5 a
- Documentation improvements to tutorials a
- serialisation of date time value for ``ReminderScheduled`` event a
 a
.. _corev0-8-0: a
 a
[0.8.0] - 2018-01-30 a
^^^^^^^^^^^^^^^^^^^^ a
 a
This is a major version change. Make sure to take a look at the a
:ref:`migration-guide` in the documentation for advice on how to a
update existing projects. a
 a
Added a
----- a
- ``--debug`` and ``--verbose`` flags to scripts (train.py, run.py, server.py) a
  to set the log level a
- support for story cycles when using checkpoints a
- added a new machine learning policy `SklearnPolicy` that uses an sklearn a
  classifier to predict actions (logistic regression by default) a
- warn if action emits events when using a model that it did never emit in a
  any of the stories the model was trained on a
- support for event pushing and endpoints to retrieve the tracker state from the server a
- Timestamp to every event a
- added a Slack channel that allows Rasa Core to communicate via a Slack app a
- added a Telegram channel that allows Rasa Core to communicate via a Telegram bot a
 a
Changed a
------- a
- rewrite of the whole FB connector: replaced pymessenger library with fbmessenger a
- story file utterance format changed from ``* _intent_greet[name=Rasa]`` a
  to ``* intent_greet{"name": "Rasa"}`` (old format is still supported but a
  deprecated) a
- persist action names in domain during model persistence a
- improved travis build speed by not using miniconda a
- don't fail with an exception but with a helpful error message if an a
  utterance template contains a variable that can not be filled a
- domain doesn't fail on unknown actions but emits a warning instead. this is to support reading a
  logs from older conversation if one recently removed an action from the domain a
 a
Fixed a
----- a
- proper evaluation of stories with checkpoints a
- proper visualisation of stories with checkpoints a
- fixed float slot min max value handling a
- fixed non integer feature decoding, e.g. used for memoization policy a
- properly log to specified file when starting Rasa Core server a
- properly calculate offset of last reset event after loading tracker from a
  tracker store a
- UserUtteranceReverted action incorrectly triggered actions to be replayed a
 a
 a
[0.7.9] - 2017-11-29 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- visualisation using Networkx version 2.x a
- add output about line of failing intent when parsing story files a
 a
[0.7.8] - 2017-11-27 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Pypi readme rendering a
 a
[0.7.7] - 2017-11-24 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- log bot utterances to tracker a
 a
Fixed a
----- a
- documentation improvements in README a
- renamed interpreter argument to rasa core server a
 a
[0.7.6] - 2017-11-15 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- moodbot example train command in docs a
 a
 a
[0.7.5] - 2017-11-14 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- "sender_id" (and "DEFAULT_SENDER_ID") keyword consistency issue #56 a
 a
Fixed a
----- a
- improved moodbot example - more nlu examples as well as better fitting of dialogue model a
 a
 a
[0.7.4] - 2017-11-09 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
 a
- added method to tracker to retrieve the latest entities #68 a
 a
[0.7.3] - 2017-10-31 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- parameter to specify font size when rendering story visualization a
 a
Fixed a
----- a
- fixed documentation of story visualization a
 a
[0.7.2] - 2017-10-30 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- added facebook bot example a
- added support for conditional checkpoints. a checkpoint can be restricted to a
  only allow one to use it if certain slots are set. see docs for details a
- utterance templates in domain yaml support buttons and images a
- validate domain yaml and raise exception on invalid file a
- ``run`` script to load models and handle messages from an input channel a
 a
Changed a
------- a
- small dropout in standard keras model to decrease reliance on exact intents a
- a LOT of documentation improvements a
 a
Fixed a
----- a
- fixed http error if action listen is not confirmed. #42 a
 a
[0.7.1] - 2017-10-06 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- issues with restart events. They created wrong a messed up history leading to a
  wrong predictions a
 a
 a
.. _corev0-7-0: a
 a
[0.7.0] - 2017-10-04 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- support for Rasa Core usage as a server with remote action execution a
 a
Changed a
------- a
- switched to max code line length 80 a
- removed action id - use ``action.name()`` instead. if an action implementation overrides the name, it should include the ``action_`` prefix (as it is not automatically added anymore) a
- renamed ``rasa_dm.util`` to ``rasa_dm.utils`` a
- renamed the whole package to ``rasa_core`` (so ``rasa_dm`` is gone!) a
- renamed ``Reminder`` attribute ``id`` to ``name`` a
- a lot of documentation improvements. docs are now at https://rasa.com/docs/core a
- use hashing when writing memorized turns into persistence - requires retraining of all models that are trained with a version prior to this a
- changed ``agent.handle_message(...)`` interface for easier usage a
 a
.. _corev0-6-0: a
 a
[0.6.0] - 2017-08-27 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- support for multiple policies (e.g. one memoization and a Keras policy at the same time) a
- loading domains from yaml files instead of defining them with python code a
- added an api layer (called ``Agent``) for you to use for 95% of the things you want to do (training, persistence, loading models) a
- support for reminders a
 a
Changed a
------- a
- large refactoring of code base a
 a
.. _corev0-5-0: a
 a
[0.5.0] - 2017-06-18 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- ``ScoringPolicy`` added to policy implementations (less strict than standard default policy) a
- ``RasaNLUInterpreter`` to run a nlu instance within dm (instead of using the http interface) a
- more tests a
 a
Changed a
------- a
- ``UserUtterance`` now holds the complete parse data from nlu (e.g. to access attributes other than entities or intent) a
- ``Turn`` has a reference to a ``UserUtterance`` instead of directly storing intent & entities (allows access to other data) a
- Simplified interface of output channels a
- order of actions in the DefaultPolicy in ``possible_actions`` (``ActionListen`` now always has index 0) a
 a
Fixed a
----- a
- ``RedisTrackerStore`` checks if tracker is stored before accessing it (otherwise a ``None`` access exception is thrown) a
- ``RegexInterpreter`` checks if the regex actually matches the message instead of assuming it always does a
- ``str`` implementation for all events a
- ``Controller`` can be started without an input channel (e.g. messages need to be fed into the queue manually) a
 a
.. _corev0-2-0: a
 a
[0.2.0] - 2017-05-18 a
^^^^^^^^^^^^^^^^^^^^ a
First released version. a
 a
 a
.. _`master`: https://github.com/RasaHQ/rasa_core/ a
 a
.. _`Semantic Versioning`: http://semver.org/ a
 a