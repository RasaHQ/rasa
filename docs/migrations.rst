:desc: Upgrading your Rasa Core Project

.. _migration:

Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.


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


0.8.x to 0.9.0
--------------

.. warning::

  This is a release **breaking backwards compatibility**.
  Unfortunately, it is not possible to load
  previously trained models (as the stored file formats have changed as
  well as the configuration and metadata). Please make sure to retrain
  a model before trying to use it with this improved version.

- loading data should be done either using

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


0.7.x to 0.8.0
--------------

- Credentials for the facebook connector changed. Instead of providing

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
  deprecated). Instead of writing

  .. code-block:: md

      ## story_07715946                     <!-- name of the story - just for debugging -->
      * _greet
         - action_ask_howcanhelp
      * _inform[location=rome,price=cheap]
         - action_on_it                     <!-- user utterance, in format _intent[entities] -->
         - action_ask_cuisine

  The new format looks like this:

  .. code-block:: md

      ## story_07715946                     <!-- name of the story - just for debugging -->
      * greet
         - action_ask_howcanhelp
      * inform{"location": "rome", "price": "cheap"}
         - action_on_it                     <!-- user utterance, in format _intent[entities] -->
         - action_ask_cuisine
