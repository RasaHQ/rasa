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



Data Augmentation and Max History
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

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


Embedding Policy
^^^^^^^^^^^^^^^^

The embedding policy is based on machine learning, and tries to learn
which actions are similar to others. 

TODO
