:desc: Information about changes between major versions of chatbot framework a
       Rasa Core and how you can migrate from one version to another. a
 a
.. _old-core-migration-guide: a
 a
Migration Guide a
=============== a
This page contains information about changes between major versions and a
how you can migrate from one version to another. a
 a
.. _migration-to-0-14-0: a
 a
0.13.x to 0.14.0 a
 a
General a
~~~~~~~ a
 a
- The python package has a new name, as does the module. You should install a
  the package using ``pip install rasa`` (instead of ``rasa_core``). a
 a
  The code moved from ``rasa_core`` to ``rasa.core`` - best way to fix is a a
  search and replace for the two most common usages: a
  ``from rasa_core`` and ``import rasa_core``. a
 a
  We have added a backwards compatibility package to still allow you to import a
  from ``rasa_core``, this will emit a warning but all imports will still a
  work. Nevertheless, you should do the above renaming of any access a
  to ``rasa_core``. a
 a
-The `MappingPolicy` is now included in `default_config.yml`. If you are using a
  a custom policy configuration make sure to update it appropriately. a
 a
- deprecated ``remote.py`` got removed - the API should be consumed directly a
  instead or with the help of the ``rasa_core_sdk``. a
 a
Asynchronous First a
~~~~~~~~~~~~~~~~~~ a
- **No more flask.** The flask webserver has been replaced with an asyncronous a
  webserver called Sanic. If you run the server in production using a wsgi a
  runner, there are instructions here on how to recreate that with the a
  sanic webserver: a
  https://sanic.readthedocs.io/en/latest/sanic/deploying.html#running-via-gunicorn a
- **Agent**: some of the method signatures changed from normal functions to a
  async coroutines. These functions need to be awaited when called, e.g. a
  ``await agent.handle_message(...)``. Changed functions include a
  - ``handle_message`` a
  - ``handle_text`` a
  - ``log_message`` a
  - ``execute_action`` a
  - ``load_data`` a
  - ``visualize`` a
 a
Custom Input / Output Channels a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
If you wrote your own input output channels, there are a couple of changes a
necessary to make the channels work properly with the asyncio server operation: a
 a
- **Need to provide Sanic blueprints.** To make the server fully asynchronous a
  the input channels need to provide Sanic blueprints instead of flask a
  blueprints. Imports should change from a
  ``from flask import Blueprint, request`` to a
  ``from sanic import Blueprint, response``. All route functions, e.g. a
  ``def webhook(...)`` need to be async and accept a request parameter as a
  their first argument, e.g. ``async def webhook(request, ...)``. a
 a
  Calls to ``on_new_message(...)`` need to be awaited: a
  ``await on_new_message(...)``. a
 a
  All output channel functions need to be async: a
  ``send_text_message``, ``send_image_url``, ``send_attachment``, a
  ``send_response``, ``send_text_with_buttons`` and ``send_custom_message``. a
  And all internal calls to these methods need to be awaited. a
 a
  For inspiration, feel free to check the code of the existing channels. a
 a
Function Naming a
~~~~~~~~~~~~~~~ a
- renamed ``train_dialogue_model`` to ``train``. Please use ``train`` from a
  now on. a
- renamed ``rasa_core.evaluate`` to ``rasa_core.test``. Please use ``test`` a
  from now on. a
 a
.. _migration-to-0-13-0: a
 a
0.12.x to 0.13.0 a
---------------- a
 a
.. warning:: a
 a
    Python 2 support has now been completely dropped: to upgrade to a
    this version you **must use Python 3**.  As always, **make sure** a
    **you retrain your models when switching to this version** a
 a
General a
~~~~~~~ a
 a
- Support for Python 2 has now been completely removed from Rasa Core, please a
  upgrade to Python 3.5 or 3.6 to continue using the software a
- If you were using the deprecated intent/entity format (``_intent[entity1=val1, entity=val2]``), a
  then you will have to update your training data to the standard format a
  (``/intent{"entity1": val1, "entity2": val2``} because it is no longer supported a
 a
.. _migration-to-0-12-0: a
 a
0.11.x to 0.12.0 a
---------------- a
 a
.. warning:: a
 a
    This is major new version with a lot of changes under the hood as well a
    as on the API level. Please take a careful look at the mentioned a
    before updating. Please make sure to a
    **retrain your models when switching to this version**. a
 a
Train script a
~~~~~~~~~~~~ a
 a
- You **must** pass a policy config flag with ``-c/--config`` now when training a
  a model, see :ref:`policy_file`. a
- Interactive learning is now started with a
  ``python -m rasa_core.train interactive`` rather than the a
  ``--interactive`` flag a
- All policy configuration related flags have been removed (``--epochs``, a
  ``--max_history``, ``--validation_split``, ``--batch_size``, a
  ``--nlu_threshold``, ``--core_threshold``, a
  ``--fallback_action_name``), specify these in the policy config file instead, a
  see :ref:`policy_file` a
 a
Visualisation script a
~~~~~~~~~~~~~~~~~~~~ a
 a
- You **must** pass a policy config flag with ``-c/--config`` now, a
  see :ref:`policy_file`. a
 a
Evaluation script a
~~~~~~~~~~~~~~~~~ a
 a
- The ``--output`` flag now takes one argument: the name of the folder a
  any files generated from the script should be written to a
- The ``--failed`` flag was removed, as this is part of the ``--output`` a
  flag now a
 a
Forms a
~~~~~ a
 a
- Forms were completely reworked, please follow :ref:`forms` a
  for instructions how to use them. a
- ``FormField`` class and its subclasses were removed, a
  overwrite ``FormAction.slot_mapping()`` method to specify the mapping between a
  user input and requested slot in the form a
  utilizing helper methods ``FormAction.from_entity(...)``, a
  ``FormAction.from_intent(...)`` and ``FormAction.from_text(...)`` a
- stories for forms need to be written differently, a
  it is recommended to use interactive learning to create form stories a
- functionality of ``FormAction.get_other_slots(...)`` was moved to a
  ``FormAction.extract_other_slots(...)`` a
- functionality of ``FormAction.get_requested_slot(...)`` was moved to a
  ``FormAction.extract_requested_slot(...)`` a
- overwrite ``FormAction.validate(...)`` method to validate user input against a
  the slot requested by the form a
 a
.. _migration-to-0-11-0: a
 a
0.10.x to 0.11.0 a
---------------- a
 a
.. warning:: a
 a
    This is major new version with a lot of changes under the hood as well a
    as on the API level. Please take a careful look at the mentioned a
    before updating. Please make sure to a
    **retrain your models when switching to this version**. a
 a
General a
~~~~~~~ a
.. note:: a
 a
  TL;DR these are the most important surface changes. But if you have a
  a second please take a minute to read all of them. a
 a
- If you have custom actions, you now need to run a separate server to execute a
  them. If your actions are written in python (in a file called actions.py) you a
  can do this by running ``python -m rasa_core_sdk.endpoint --actions actions`` a
  and specifying the action endpoint in the ``endpoints.yml`` a
  For more information please read :ref:`custom actions <custom-actions>`. a
- For your custom actions, the imports have changed from a
  ``from rasa_core.actions import Action`` to ``from rasa_core_sdk import Action`` and a
  from ``from rasa_core.events import *`` to ``from rasa_core_sdk.events import *`` a
- The actions list in the domain now needs to always contain the actions names a
  instead of the classpath (e.g. change ``actions.ActionExample`` to ``action_example``) a
- utter templates that should be used as actions, now need to start with a
  ``utter_``, otherwise the bot won't be able to find the action a
 a
HTTP Server endpoints a
~~~~~~~~~~~~~~~~~~~~~ a
- We removed ``/parse`` and ``/continue`` endpoints used for running actions a
  remotely. This has been replaced by the action server that allows you a
  to run your action code in any language. There are no replacement endpoints a
  for these two, as the flow of information has been changed: Instead of you a
  calling Rasa Core to update the tracker and receive the next action to be a
  executed, Rasa Core will call your action server once it predicted an action. a
  More information can be found in the updated docs for :ref:`custom actions <custom-actions>`. a
 a
 a
Webhooks a
~~~~~~~~ a
- The endpoints for the webhooks changed. All webhooks are now at a
  ``/webhooks/CHANNEL_NAME/webhook``. For example, the webhook a
  to receive facebook messages on a local instance is now a
  ``http://localhost:5005/webhooks/facebook/webhook``. a
- format of the ``credentials.yml`` used in the ``run`` and ``server`` scripts a
  has changed to allow for multiple channels in one file: a
 a
  The new format now contains the channels name first, e.g. for facebook: a
 a
  .. code-block:: yaml a
 a
     facebook: a
       verify: "rasa-bot" a
       secret: "3e34709d01ea89032asdebfe5a74518" a
       page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD" a
 a
Changes to Input and Output Channels a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
- ``ConsoleOutputChannel`` and ``ConsoleInputChannel`` have been removed. Either a
  use the `run script <https://github.com/RasaHQ/rasa_core/blob/master/rasa_core/run.py>`_ a
  to run your bot on the cmdline, or adapt the ``serve_application`` a
  `function <https://github.com/RasaHQ/rasa_core/blob/master/rasa_core/run.py#L260>`_ a
  to run from a python script. a
- ``rasa_core.channels.direct`` output channel package removed. a
  ``CollectingOutputChannel`` moved to ``rasa_core.channels.channel`` a
- ``HttpInputComponent`` renamed to ``InputChannel`` & moved to a
  ``rasa_core.channels.channel.InputChannel`` a
- If you wrote your own custom input channel, make sure to inherit from a
  ``InputChannel`` instead of ``HttpInputComponent``. a
- ``CollectingOutput`` channel will no properly collect events for images, a
  buttons, and attachments. The content of the collected messages has changed, a
  ``data`` is now called ``buttons``. a
- removed package ``rasa_core.channels.rest``, a
  please use ``rasa_core.channels.RestInput`` instead a
- remove file input channel ``rasa_core.channels.file.FileInputChannel`` a
- signature of ``agent.handle_channel`` got renamed a
  and the signature changed. here is an up to date example: a
 a
  .. code-block:: python a
 a
     from rasa_core.channels.facebook import FacebookInput a
 a
     input_channel = FacebookInput(fb_verify="VERIFY", a
                                   fb_secret="SECRET", a
                                   fb_access_token="ACCESS_TOKEN") a
     agent.handle_channels([input_channel], port=5005, serve_forever=True) a
- If you wrote your own custom output channel, make sure to split messages a
  on double new lines if you like (the ``InputChannel`` you inherit from a
  doesn't do this anymore), e.g.: a
 a
  .. code-block:: python a
 a
     def send_text_message(self, recipient_id: Text, message: Text) -> None: a
         """Send a message through this channel.""" a
 a
         for message_part in message.split("\n\n"): a
           # self.send would be the actual communication to e.g. facebook a
           self.send(recipient_id, message_part) a
 a
 a
.. _migration-to-0-10-0: a
 a
0.9.x to 0.10.0 a
--------------- a
.. warning:: a
 a
  This is a release **breaking backwards compatibility**. a
  You can no longer load old models with this version, due to the addition of a
  the default action ``ActionDefaultFallback``. Please make sure to retrain a
  your model before using this version a
 a
There have been some API changes to classes and methods: a
 a
- if you use ``dispatcher.utter_template`` or a
  ``dispatcher.utter_button_template`` in your custom actions run code, a
  they now need the ``tracker`` as a second argument, e.g. a
  ``dispatcher.utter_template("utter_greet", tracker)`` a
 a
- all input and output channels should have a ``name``. If you are using a a
  custom channel, make sure to implement a class method that returns a
  the name. The name needs to be added to the a
  **input channel and the output channel**. You can find examples a
  in ``rasa_core.channels.direct.CollectingOutputChannel``: a
 a
  .. code-block:: python a
 a
      @classmethod a
      def name(cls): a
          """Every channel needs a name""" a
          return "collector" a
 a
- the ``RasaNLUHttpInterpreter`` when created now needs to be passed an a
  instance of ``EndpointConfig`` instead of ``server`` and ``token``, e.g.: a
 a
  .. code-block:: python a
 a
      from rasa_core.utils import EndpointConfig a
 a
      endpoint = EndpointConfig("http://localhost:500", token="mytoken") a
      interpreter = RasaNLUHttpInterpreter("mymodelname", endpoint) a
 a
.. _migration-to-0-9-0: a
 a
0.8.x to 0.9.0 a
-------------- a
 a
.. warning:: a
 a
  This is a release **breaking backwards compatibility**. a
  Unfortunately, it is not possible to load a
  previously trained models (as the stored file formats have changed as a
  well as the configuration and metadata). Please make sure to retrain a
  a model before trying to use it with this improved version. a
 a
- loading data should be done either using: a
 a
  .. code-block:: python a
 a
      from rasa_core import training a
 a
      training_data = training.load_data(...) a
 a
  or using an agent instance: a
 a
  .. code-block:: python a
 a
      training_data = agent.load_data(...) a
      agent.train(training_data, ...) a
 a
  It is deprecated to pass the training data file directly to ``agent.train``. a
  Instead, the data should be loaded in one of the above ways and then passed a
  to train. a
 a
- ``ScoringPolicy`` got removed and replaced by ``AugmentedMemoizationPolicy`` a
  which is similar, but is able to match more states to states it has seen a
  during trainer (e.g. it is able to handle slots better) a
 a
- if you use custom featurizers, you need to a
  **pass them directly to the policy** that should use them. a
  This allows the policies to use different featurizers. Passing a featurizer a
  is **optional**. Accordingly, the ``max_history`` parameter moved to that a
  featurizer: a
 a
  .. code-block:: python a
 a
      from rasa_core.featurizers import (MaxHistoryTrackerFeaturizer, a
                                         BinarySingleStateFeaturizer) a
 a
      featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(), a
                                               max_history=5) a
 a
      agent = Agent(domain_file, a
                    policies=[MemoizationPolicy(max_history=5), a
                              KerasPolicy(featurizer)]) a
 a
  If no featurizer is passed during policy creation, the policies default a
  featurizer will be used. The `MemoizationPolicy` allows passing in the a
  `max_history` parameter directly, without creating a featurizer. a
 a
- the ListSlot now stores a list of entities (with the same name) a
  present in an utterance a
 a
 a
.. _migration-to-0-8-0: a
 a
0.7.x to 0.8.0 a
-------------- a
 a
- Credentials for the facebook connector changed. Instead of providing: a
 a
  .. code-block:: yaml a
 a
      # OLD FORMAT a
      verify: "rasa-bot" a
      secret: "3e34709d01ea89032asdebfe5a74518" a
      page-tokens: a
        1730621093913654: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD" a
 a
  you should now pass the configuration parameters like this: a
 a
  .. code-block:: yaml a
 a
      # NEW FORMAT a
      verify: "rasa-bot" a
      secret: "3e34709d01ea89032asdebfe5a74518" a
      page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD" a
 a
  As you can see, the new facebook connector only supports a single page. Same a
  change happened to the in code arguments for the connector which should be a
  changed to: a
 a
  .. code-block:: python a
 a
      from rasa_core.channels.facebook import FacebookInput a
 a
      FacebookInput( a
            credentials.get("verify"), a
            credentials.get("secret"), a
            credentials.get("page-access-token")) a
 a
- Story file format changed from ``* _intent_greet[name=Rasa]`` a
  to ``* intent_greet{"name": "Rasa"}`` (old format is still supported but a
  deprecated). Instead of writing: a
 a
  .. code-block:: story a
 a
      ## story_07715946                     <!-- name of the story - just for debugging --> a
      * _greet a
         - action_ask_howcanhelp a
      * _inform[location=rome,price=cheap] a
         - action_on_it                     <!-- user utterance, in format _intent[entities] --> a
         - action_ask_cuisine a
 a
  The new format looks like this: a
 a
  .. code-block:: story a
 a
      ## story_07715946                     <!-- name of the story - just for debugging --> a
      * greet a
         - action_ask_howcanhelp a
      * inform{"location": "rome", "price": "cheap"} a
         - action_on_it                     <!-- user utterance, in format _intent[entities] --> a
         - action_ask_cuisine a
 a