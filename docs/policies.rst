.. _custom_policies:

Custom Policies
===============


The ``Policy`` is the core of your bot, and it really just has one important method:

.. doctest::

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]

        return []


This uses the current state of the conversation (provided by the tracker) to choose the next action to take.
The domain is there if you need it, but only some policy types make use of it. The returned array contains
the probabilities for each action to be executed next. The action that is most likely will be executed.


Let's look at a simple example for a custom policy:

.. doctest::

  from rasa_core.policies import Policy
  from rasa_core.actions.action import ACTION_LISTEN_NAME
  from rasa_core import utils
  import numpy as np

  class SimplePolicy(Policy):
      def predict_action_probabilities(self, tracker, domain):
          responses = {"greet": 3}

          if tracker.latest_action_name == ACTION_LISTEN_NAME:
              key = tracker.latest_message.intent["name"]
              action = responses[key] if key in responses else 2
              return utils.one_hot(action, domain.num_actions)
          else:
              return np.zeros(domain.num_actions)


**How does this work?**
When the controller processes a message from a user, it will keep asking for the next most likely action using ``predict_action_probabilities``.
The bot then executes that action, until it receives an ``ActionListen`` instruction.
This breaks the loop and makes the bot await further instructions. 

In pseudocode, what the ``SimplePolicy`` above does is:

.. code-block:: md

    -> a new message has come in

    if we were previously listening:
        return a canned response
    else:
        we must have just said something, so let's Listen again


Note that the policy itself is stateless, and all the state is carried by the ``tracker`` object.


Creating Policies from Stories
------------------------------

Writing rules like in the SimplePolicy above is not a great way to build a bot, it gets messy fast & is hard to debug.
If you've found Rasa Core, it's likely you've already tried this approach and were looking for something better.
A good next step is to use our story framework to build a policy by giving it some example conversations.
We won't use machine learning yet, we will just create a policy which memorises these stories. 

We can use the ``MemoizationPolicy`` and the ``PolicyTrainer`` classes to do this.

Here is the ``PolicyTrainer`` class:

.. literalinclude:: ../rasa_core/policies/trainer.py
   :pyobject: PolicyTrainer

What the ``train()`` method does is the following:

1. reads the stories from a file
2. creates all possible dialogues from these stories
3. creates the following variables:

   a. ``y`` - a 1D array representing all of the actions taken in the dialogues
   b. ``X`` - a 2D array where each row represents the state of the tracker when an action was taken

4. calls the policy's ``train()`` method to create a policy from these ``X, y``
   state-action pairs (don't mind the ``ensemble`` it is just a collection of
   policies - e.g. you can combine multiple policies and train them all at
   once using the ensemble)


.. note::

    In fact, the rows in ``X`` describe the state of the tracker when the previous ``max_history`` actions were taken. See :ref:`featurization` for more details.

For the ``MemoizationPolicy``, the ``train()`` method just memorises the actions taken in the story,
so that when your bot encounters an identical situation it will make the decision you intended. 


Generalising to new Dialogues
-----------------------------

The stories data format gives you a compact way to describe a large number of possible dialogues without much effort. 
But humans are infinitely creative, and you could never hope to describe *every* possible dialogue programatically.
Even if you could, it probably wouldn't fit in memory :)

So how do we create a policy which behaves well even in scenarios you haven't thought of?
We will try to achieve this generalisation by creating a policy based on Machine Learning. 

You can use whichever machine learning library you like to train your policy.
One implementation that ships with Rasa is the ``KerasPolicy``,
which uses Keras as a machine learning library to train your dialogue model.
These base classes have already implemented the logic of persisting and reloading models.

By default, each of these trains a linear model to fit the ``X, y`` data.

The model is defined here: 

.. literalinclude:: ../rasa_core/policies/keras_policy.py
   :pyobject: KerasPolicy.model_architecture


and the training is run here:

.. literalinclude:: ../rasa_core/policies/keras_policy.py
   :pyobject: KerasPolicy.train


You can implement the model of your choice by overriding these methods.