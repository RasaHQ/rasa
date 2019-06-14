:desc: Information about changes between major versions of chatbot framework
       Rasa Core and how you can migrate from one version to another.

.. _old-core-migration-guide:

Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

.. _migration-to-0-14-0:

0.13.x to 0.14.0

General
~~~~~~~

- The python package has a new name, as does the module. You should install
  the package using ``pip install rasa`` (instead of ``rasa_core``).

  The code moved from ``rasa_core`` to ``rasa.core`` - best way to fix is a
  search and replace for the two most common usages:
  ``from rasa_core`` and ``import rasa_core``.

  We have added a backwards compatibility package to still allow you to import
  from ``rasa_core``, this will emit a warning but all imports will still
  work. Nevertheless, you should do the above renaming of any access
  to ``rasa_core``.

-The `MappingPolicy` is now included in `default_config.yml`. If you are using
  a custom policy configuration make sure to update it appropriately.

- deprecated ``remote.py`` got removed - the API should be consumed directly
  instead or with the help of the ``rasa_core_sdk``.

Asynchronous First
~~~~~~~~~~~~~~~~~~
- **No more flask.** The flask webserver has been replaced with an asyncronous
  webserver called Sanic. If you run the server in production using a wsgi
  runner, there are instructions here on how to recreate that with the
  sanic webserver:
  https://sanic.readthedocs.io/en/latest/sanic/deploying.html#running-via-gunicorn
- **Agent**: some of the method signatures changed from normal functions to
  async coroutines. These functions need to be awaited when called, e.g.
  ``await agent.handle_message(...)``. Changed functions include
  - ``handle_message``
  - ``handle_text``
  - ``log_message``
  - ``execute_action``
  - ``load_data``
  - ``visualize``

Custom Input / Output Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you wrote your own input output channels, there are a couple of changes
necessary to make the channels work properly with the asyncio server operation:

- **Need to provide Sanic blueprints.** To make the server fully asynchronous
  the input channels need to provide Sanic blueprints instead of flask
  blueprints. Imports should change from
  ``from flask import Blueprint, request`` to
  ``from sanic import Blueprint, response``. All route functions, e.g.
  ``def webhook(...)`` need to be async and accept a request parameter as
  their first argument, e.g. ``async def webhook(request, ...)``.

  Calls to ``on_new_message(...)`` need to be awaited:
  ``await on_new_message(...)``.

  All output channel functions need to be async:
  ``send_text_message``, ``send_image_url``, ``send_attachment``,
  ``send_response``, ``send_text_with_buttons`` and ``send_custom_message``.
  And all internal calls to these methods need to be awaited.

  For inspiration, feel free to check the code of the existing channels.

Function Naming
~~~~~~~~~~~~~~~
- renamed ``train_dialogue_model`` to ``train``. Please use ``train`` from
  now on.
- renamed ``rasa_core.evaluate`` to ``rasa_core.test``. Please use ``test``
  from now on.

.. _migration-to-0-13-0:

0.12.x to 0.13.0
----------------

.. warning::

    Python 2 support has now been completely dropped: to upgrade to
    this version you **must use Python 3**.  As always, **make sure**
    **you retrain your models when switching to this version**

General
~~~~~~~

- Support for Python 2 has now been completely removed from Rasa Core, please
  upgrade to Python 3.5 or 3.6 to continue using the software
- If you were using the deprecated intent/entity format (``_intent[entity1=val1, entity=val2]``),
  then you will have to update your training data to the standard format
  (``/intent{"entity1": val1, "entity2": val2``} because it is no longer supported

.. _migration-to-0-12-0:

0.11.x to 0.12.0
----------------

.. warning::

    This is major new version with a lot of changes under the hood as well
    as on the API level. Please take a careful look at the mentioned
    before updating. Please make sure to
    **retrain your models when switching to this version**.

Train script
~~~~~~~~~~~~

- You **must** pass a policy config flag with ``-c/--config`` now when training
  a model, see :ref:`policy_file`.
- Interactive learning is now started with
  ``python -m rasa_core.train interactive`` rather than the
  ``--interactive`` flag
- All policy configuration related flags have been removed (``--epochs``,
  ``--max_history``, ``--validation_split``, ``--batch_size``,
  ``--nlu_threshold``, ``--core_threshold``,
  ``--fallback_action_name``), specify these in the policy config file instead,
  see :ref:`policy_file`

Visualisation script
~~~~~~~~~~~~~~~~~~~~

- You **must** pass a policy config flag with ``-c/--config`` now,
  see :ref:`policy_file`.

Evaluation script
~~~~~~~~~~~~~~~~~

- The ``--output`` flag now takes one argument: the name of the folder
  any files generated from the script should be written to
- The ``--failed`` flag was removed, as this is part of the ``--output``
  flag now

Forms
~~~~~

- Forms were completely reworked, please follow :ref:`forms`
  for instructions how to use them.
- ``FormField`` class and its subclasses were removed,
  overwrite ``FormAction.slot_mapping()`` method to specify the mapping between
  user input and requested slot in the form
  utilizing helper methods ``FormAction.from_entity(...)``,
  ``FormAction.from_intent(...)`` and ``FormAction.from_text(...)``
- stories for forms need to be written differently,
  it is recommended to use interactive learning to create form stories
- functionality of ``FormAction.get_other_slots(...)`` was moved to
  ``FormAction.extract_other_slots(...)``
- functionality of ``FormAction.get_requested_slot(...)`` was moved to
  ``FormAction.extract_requested_slot(...)``
- overwrite ``FormAction.validate(...)`` method to validate user input against
  the slot requested by the form

.. _migration-to-0-11-0:

0.10.x to 0.11.0
----------------

.. warning::

    This is major new version with a lot of changes under the hood as well
    as on the API level. Please take a careful look at the mentioned
    before updating. Please make sure to
    **retrain your models when switching to this version**.

General
~~~~~~~
.. note::

  TL;DR these are the most important surface changes. But if you have
  a second please take a minute to read all of them.

- If you have custom actions, you now need to run a separate server to execute
  them. If your actions are written in python (in a file called actions.py) you
  can do this by running ``python -m rasa_core_sdk.endpoint --actions actions``
  and specifying the action endpoint in the ``endpoints.yml``
  For more information please read :ref:`custom actions <custom-actions>`.
- For your custom actions, the imports have changed from
  ``from rasa_core.actions import Action`` to ``from rasa_core_sdk import Action`` and
  from ``from rasa_core.events import *`` to ``from rasa_core_sdk.events import *``
- The actions list in the domain now needs to always contain the actions names
  instead of the classpath (e.g. change ``actions.ActionExample`` to ``action_example``)
- utter templates that should be used as actions, now need to start with
  ``utter_``, otherwise the bot won't be able to find the action

HTTP Server endpoints
~~~~~~~~~~~~~~~~~~~~~
- We removed ``/parse`` and ``/continue`` endpoints used for running actions
  remotely. This has been replaced by the action server that allows you
  to run your action code in any language. There are no replacement endpoints
  for these two, as the flow of information has been changed: Instead of you
  calling Rasa Core to update the tracker and receive the next action to be
  executed, Rasa Core will call your action server once it predicted an action.
  More information can be found in the updated docs for :ref:`custom actions <custom-actions>`.


Webhooks
~~~~~~~~
- The endpoints for the webhooks changed. All webhooks are now at
  ``/webhooks/CHANNEL_NAME/webhook``. For example, the webhook
  to receive facebook messages on a local instance is now
  ``http://localhost:5005/webhooks/facebook/webhook``.
- format of the ``credentials.yml`` used in the ``run`` and ``server`` scripts
  has changed to allow for multiple channels in one file:

  The new format now contains the channels name first, e.g. for facebook:

  .. code-block:: yaml

     facebook:
       verify: "rasa-bot"
       secret: "3e34709d01ea89032asdebfe5a74518"
       page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"

Changes to Input and Output Channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``ConsoleOutputChannel`` and ``ConsoleInputChannel`` have been removed. Either
  use the `run script <https://github.com/RasaHQ/rasa_core/blob/master/rasa_core/run.py>`_
  to run your bot on the cmdline, or adapt the ``serve_application``
  `function <https://github.com/RasaHQ/rasa_core/blob/master/rasa_core/run.py#L260>`_
  to run from a python script.
- ``rasa_core.channels.direct`` output channel package removed.
  ``CollectingOutputChannel`` moved to ``rasa_core.channels.channel``
- ``HttpInputComponent`` renamed to ``InputChannel`` & moved to
  ``rasa_core.channels.channel.InputChannel``
- If you wrote your own custom input channel, make sure to inherit from
  ``InputChannel`` instead of ``HttpInputComponent``.
- ``CollectingOutput`` channel will no properly collect events for images,
  buttons, and attachments. The content of the collected messages has changed,
  ``data`` is now called ``buttons``.
- removed package ``rasa_core.channels.rest``,
  please use ``rasa_core.channels.RestInput`` instead
- remove file input channel ``rasa_core.channels.file.FileInputChannel``
- signature of ``agent.handle_channel`` got renamed
  and the signature changed. here is an up to date example:

  .. code-block:: python

     from rasa_core.channels.facebook import FacebookInput

     input_channel = FacebookInput(fb_verify="VERIFY",
                                   fb_secret="SECRET",
                                   fb_access_token="ACCESS_TOKEN")
     agent.handle_channels([input_channel], port=5005, serve_forever=True)
- If you wrote your own custom output channel, make sure to split messages
  on double new lines if you like (the ``InputChannel`` you inherit from
  doesn't do this anymore), e.g.:

  .. code-block:: python

     def send_text_message(self, recipient_id: Text, message: Text) -> None:
         """Send a message through this channel."""

         for message_part in message.split("\n\n"):
           # self.send would be the actual communication to e.g. facebook
           self.send(recipient_id, message_part)


.. _migration-to-0-10-0:

0.9.x to 0.10.0
---------------
.. warning::

  This is a release **breaking backwards compatibility**.
  You can no longer load old models with this version, due to the addition of
  the default action ``ActionDefaultFallback``. Please make sure to retrain
  your model before using this version

There have been some API changes to classes and methods:

- if you use ``dispatcher.utter_template`` or
  ``dispatcher.utter_button_template`` in your custom actions run code,
  they now need the ``tracker`` as a second argument, e.g.
  ``dispatcher.utter_template("utter_greet", tracker)``

- all input and output channels should have a ``name``. If you are using a
  custom channel, make sure to implement a class method that returns
  the name. The name needs to be added to the
  **input channel and the output channel**. You can find examples
  in ``rasa_core.channels.direct.CollectingOutputChannel``:

  .. code-block:: python

      @classmethod
      def name(cls):
          """Every channel needs a name"""
          return "collector"

- the ``RasaNLUHttpInterpreter`` when created now needs to be passed an
  instance of ``EndpointConfig`` instead of ``server`` and ``token``, e.g.:

  .. code-block:: python

      from rasa_core.utils import EndpointConfig

      endpoint = EndpointConfig("http://localhost:500", token="mytoken")
      interpreter = RasaNLUHttpInterpreter("mymodelname", endpoint)

.. _migration-to-0-9-0:

0.8.x to 0.9.0
--------------

.. warning::

  This is a release **breaking backwards compatibility**.
  Unfortunately, it is not possible to load
  previously trained models (as the stored file formats have changed as
  well as the configuration and metadata). Please make sure to retrain
  a model before trying to use it with this improved version.

- loading data should be done either using:

  .. code-block:: python

      from rasa_core import training

      training_data = training.load_data(...)

  or using an agent instance:

  .. code-block:: python

      training_data = agent.load_data(...)
      agent.train(training_data, ...)

  It is deprecated to pass the training data file directly to ``agent.train``.
  Instead, the data should be loaded in one of the above ways and then passed
  to train.

- ``ScoringPolicy`` got removed and replaced by ``AugmentedMemoizationPolicy``
  which is similar, but is able to match more states to states it has seen
  during trainer (e.g. it is able to handle slots better)

- if you use custom featurizers, you need to
  **pass them directly to the policy** that should use them.
  This allows the policies to use different featurizers. Passing a featurizer
  is **optional**. Accordingly, the ``max_history`` parameter moved to that
  featurizer:

  .. code-block:: python

      from rasa_core.featurizers import (MaxHistoryTrackerFeaturizer,
                                         BinarySingleStateFeaturizer)

      featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(),
                                               max_history=5)

      agent = Agent(domain_file,
                    policies=[MemoizationPolicy(max_history=5),
                              KerasPolicy(featurizer)])

  If no featurizer is passed during policy creation, the policies default
  featurizer will be used. The `MemoizationPolicy` allows passing in the
  `max_history` parameter directly, without creating a featurizer.

- the ListSlot now stores a list of entities (with the same name)
  present in an utterance


.. _migration-to-0-8-0:

0.7.x to 0.8.0
--------------

- Credentials for the facebook connector changed. Instead of providing:

  .. code-block:: yaml

      # OLD FORMAT
      verify: "rasa-bot"
      secret: "3e34709d01ea89032asdebfe5a74518"
      page-tokens:
        1730621093913654: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"

  you should now pass the configuration parameters like this:

  .. code-block:: yaml

      # NEW FORMAT
      verify: "rasa-bot"
      secret: "3e34709d01ea89032asdebfe5a74518"
      page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"

  As you can see, the new facebook connector only supports a single page. Same
  change happened to the in code arguments for the connector which should be
  changed to:

  .. code-block:: python

      from rasa_core.channels.facebook import FacebookInput

      FacebookInput(
            credentials.get("verify"),
            credentials.get("secret"),
            credentials.get("page-access-token"))

- Story file format changed from ``* _intent_greet[name=Rasa]``
  to ``* intent_greet{"name": "Rasa"}`` (old format is still supported but
  deprecated). Instead of writing:

  .. code-block:: story

      ## story_07715946                     <!-- name of the story - just for debugging -->
      * _greet
         - action_ask_howcanhelp
      * _inform[location=rome,price=cheap]
         - action_on_it                     <!-- user utterance, in format _intent[entities] -->
         - action_ask_cuisine

  The new format looks like this:

  .. code-block:: story

      ## story_07715946                     <!-- name of the story - just for debugging -->
      * greet
         - action_ask_howcanhelp
      * inform{"location": "rome", "price": "cheap"}
         - action_on_it                     <!-- user utterance, in format _intent[entities] -->
         - action_ask_cuisine
