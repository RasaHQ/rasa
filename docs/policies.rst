.. _policies:

Training and Policies
=====================


Training
--------

Rasa Core works by creating training data from your stories and 
training a model on that data. 

You can run training from the command line like in the :ref:`quickstart`:


.. code-block:: bash

   python -m rasa_core.train -d domain.yml -s data/stories.md -o models/current/dialogue --epochs 200

Or by creating an agent and running the train method yourself:

.. testcode::

   from rasa_core.agent import Agent

   agent = Agent()
   data = agent.load_data("stories.md")
   agent.train(data)



Data Augmentation
^^^^^^^^^^^^^^^^^

By default, Rasa Core will create longer stories by randomly glueing together 
the ones in your stories file. This is because if you have stories like:

.. code-block:: md
   
    # thanks
    * thankyou
       - utter_youarewelcome

    # bye
    * goodbye
       - utter_goodbye


You actually want to teach your policy to **ignore** the dialogue history
when it isn't relevant and just respond with the same action no matter what happened
before. 

You can alter this behaviour with the ``--augmentation`` flag. ``--augmentation 0`` 
disables this behavior. 

In python, you can pass the ``augmentation_factor`` argument to the ``Agent.load_data`` method.

Max History
^^^^^^^^^^^

One important hyperparameter for Rasa Core policies is the ``max_history``.
This controls how much dialogue history the model looks at to decide which action
to take next. 

You can set the ``max_history`` using the training script's ``--history`` flag or 
by passing it to your policy's :class:`Featurizer`.

.. note:: 

    Only the ``MaxHistoryTrackerFeaturizer`` uses a max history, whereas the 
    ``FullDialogueTrackerFeaturizer`` always looks at the full conversation history.

As an example, let's say you have an ``out_of_scope`` intent which describes off-topic
user messages. If your bot sees this intent multiple times in a row, you might want to 
tell the user what you `can` help them with. So your story might look like this:

.. code-block:: md

   * out_of_scope
      - utter_default
   * out_of_scope
      - utter_default
   * out_of_scope
      - utter_help_message

For Rasa Core to learn this pattern, the ``max_history`` has to be `at least` ``3``. 

If you increase your ``max_history``, your model will become bigger and training will take longer.
If you have some information that should affect the dialogue very far into the future,
you should store it as a slot. Slot information is always available for every featurizer.



Training Script Options
^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: python -m rasa_core.train -h



Policies
--------

The :class:`rasa_core.policies.Policy` class decides which action to take
at every step in the conversation. 

There are different policies to choose from, and you can include multiple policies
in a single :class:`Agent`. At every turn, the policy which predicts the 
next action with the highest confidence will be used. 
You can pass a list of policies when you create an agent:

.. code-block:: python

   from rasa_core.policies.memoization import MemoizationPolicy
   from rasa_core.policies.keras_policy import KerasPolicy
   from rasa_core.agent import Agent

   agent = Agent("domain.yml",
                  policies=[MemoizationPolicy(), KerasPolicy()])


.. note::

    By default, Rasa Core uses the :class:`KerasPolicy` in combination with 
    the :class:`MemoizationPolicy`. 

Memoization Policy
^^^^^^^^^^^^^^^^^^

The :class:`MemoizationPolicy` just memorizes the conversations in your training data.
It predicts the next action with confidence ``1.0`` if this exact conversation exists in the
training data, otherwise it predicts ``None`` with confidence ``0.0``.


Keras Policy
^^^^^^^^^^^^

The ``KerasPolicy`` uses a neural network implemented in `Keras <http://keras.io>`_ to
select the next action.
The deafult architecture is based on an LSTM, but you can override the 
``KerasPolicy.model_architecture`` method to implement your own architecture. 


.. literalinclude:: ../rasa_core/policies/keras_policy.py
   :pyobject: KerasPolicy.model_architecture

