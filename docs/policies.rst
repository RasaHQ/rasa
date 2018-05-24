.. _policies:

Policies
========


The ``Policy`` is the core of your bot, with its most important method:

.. literalinclude:: ../rasa_core/policies/policy.py
   :pyobject: Policy.predict_action_probabilities

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
When the controller processes a message from a user, it will keep asking for the next most likely action using
``predict_action_probabilities``. The bot then executes that action, then call ``predict_action_probabilities`` again
with a new ``tracker``, until it receives an ``ActionListen`` instruction.
This breaks the loop and makes the bot await further instructions. 

In pseudocode, what the ``SimplePolicy`` above does is:

.. code-block:: md

    -> a new message has come in

    if we were previously listening:
        return a canned response
    else:
        we must have just said something, so let's listen again


Note that the policy itself is stateless, and all the state is carried by the ``tracker`` object.


Creating Policies from Stories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writing rules like in the SimplePolicy above is not a great way to build a bot, it gets messy fast & is hard to debug.
If you've found Rasa Core, it's likely you've already tried this approach and were looking for something better.

The second important method of any policy is ``train(...)``:

.. literalinclude:: ../rasa_core/policies/policy.py
   :pyobject: Policy.train

This method creates "some rules" for prediction depending on the training data.


Memorising the training data
----------------------------

A good next step is to use our story framework to build a policy by giving it some example conversations.
We won't use machine learning yet, we will just create a policy which memorises these stories. 

We can use the ``MemoizationPolicy`` to do this.

.. note::
    For the ``MemoizationPolicy``, the ``train()`` method just memorises
    the actions taken in the story of ``max_history`` turns, so that when your bot encounters an
    identical situation it will make the decision you intended.


Augmented memoization
---------------------

If it is needed to recall turns from training dialogues
where some ``slots`` might not be set during prediction time,
add relevant stories without such ``slots`` to training data.
E.g. reminder stories.

Since ``slots`` that are set some time in the past are
preserved in all future feature vectors until they are set
to None, this policy has a capability to recall the turns
up to ``max_history`` and less from training stories during prediction,
even if additional slots were filled in the past for current dialogue.


Generalising to new Dialogues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The stories data format gives you a compact way to describe a large number of possible dialogues without much effort. 
But humans are infinitely creative, and you could never hope to describe *every* possible dialogue programatically.
Even if you could, it probably wouldn't fit in memory :)

So how do we create a policy which behaves well even in scenarios you haven't thought of?
We will try to achieve this generalisation by creating a policy based on Machine Learning. 

Any policy should be initialized with a featurizer.
The policy's ``train`` method calls this featurizer on provided ``training_trackers`` to create ``X, y`` data,
suitable for ML algorithm (see :ref:`featurization` for details).

The method to featurize trackers is defined here:

.. literalinclude:: ../rasa_core/policies/policy.py
   :pyobject: Policy.featurize_for_training


Keras policy
------------

You can use whichever machine learning library you like to train your policy.
One implementation that ships with Rasa is the ``KerasPolicy``,
which uses Keras as a machine learning library to train your dialogue model.
This class has already implemented the logic of persisting and reloading models.

The model is defined here:

.. literalinclude:: ../rasa_core/policies/keras_policy.py
   :pyobject: KerasPolicy.model_architecture

and the training is run here:

.. literalinclude:: ../rasa_core/policies/keras_policy.py
   :pyobject: KerasPolicy.train

You can implement the model of your choice by overriding these methods,
or initialize ``KerasPolicy`` with already defined ``keras model``.


Embedding policy
----------------

This policy has predefined architecture, which comprises the following steps:
    - apply dense layers to create embeddings for user intents and entities and system actions including previous actions and slots;
    - concatenate the embeddings of previous user inputs and embeddings of previous system actions to create a memory;
    - concatenate user embeddings and slots into an attention wrapper input vector;
    - using the input vector and the previous LSTM output calculate attention probabilities over the memory
      using `NTM mechanism <https://arxiv.org/abs/1410.5401>`_;
    - feed the attention vectors and the embeddings of the slots as an input to an LSTM cell;
    - apply a dense layer to the output of the LSTM to get a recurrent embedding of a dialogue;
    - for each LSTM time step, calculate the similarity between this dialogue embedding and embedded system actions.
      This step is based on the starspace idea from: `<https://arxiv.org/abs/1709.03856>`_.

It is recommended to use ``LabelTokenizerSingleStateFeaturizer`` (see :ref:`featurization` for details).
.. note::
    This policy only works with ``FullDialogueTrackerFeaturizer``.

**Configuration**:

    Configuration parameters can be passed to ``agent.train(...)`` method.
    .. note:: Pass appropriate ``epochs`` number to ``agent.train(...)`` method,
              otherwise the policy will be trained only for ``1`` epoch. Since this is embedding based policy,
              it requires large number of epochs, which depends on complexity of the training data and whether
              attention was turned on or not.

    The main feature of this policy is **attention** mechanism over previous user input and system actions.
    **Attention is turned off by default**, in order to turn it on, configure the following parameters:
        - ``use_attention`` if ``true`` the algorithm will use attention mechanism, default ``false``;
        - ``sparse_attention`` if ``true`` ``sparsemax`` will be used instead of ``softmax`` for attention probabilities, default ``false``;
        - ``attn_shift_range`` the range of allowed location-based attention shifts, see `<https://arxiv.org/abs/1410.5401>`_ for details;
        - ``skip_cells`` if ``true`` the algorithm will add sigmoid gate to skip rnn time step for LSTM's hidden memory state.

    .. note:: Attention requires larger values of ``epochs`` and takes longer to train. But it can learn more complicated and nonlinear behaviour.

    The algorithm also has hyperparameters to control:
        - neural network's architecture:
            - ``num_hidden_layers_a`` and ``hidden_layer_size_a`` set the number of hidden layers and their sizes before embedding layer for user inputs;
            - ``num_hidden_layers_b`` and ``hidden_layer_size_b`` set the number of hidden layers and their sizes before embedding layer for system actions;
            - ``rnn_size`` set the number of units in the LSTM cell;
        - training:
            - ``batch_size`` sets the number of training examples in one forward/backward pass, the higher the batch size, the more memory space you'll need;
            - ``epochs`` sets the number of times the algorithm will see training data, where ``one epoch`` = one forward pass and one backward pass of all the training examples;
        - embedding:
            - ``embed_dim`` sets the dimension of embedding space;
            - ``mu_pos`` controls how similar the algorithm should try to make embedding vectors for correct intent labels;
            - ``mu_neg`` controls maximum negative similarity for incorrect intents;
            - ``similarity_type`` sets the type of the similarity, it should be either ``cosine`` or ``inner``;
            - ``num_neg`` sets the number of incorrect intent labels, the algorithm will minimize their similarity to the user input during training;
            - ``use_max_sim_neg`` if ``true`` the algorithm only minimizes maximum similarity over incorrect intent labels;
        - regularization:
            - ``C2`` sets the scale of L2 regularization
            - ``C_emb`` sets the scale of how important is to minimize the maximum similarity between embeddings of different intent labels;
            - ``droprate_a`` sets the dropout rate between hidden layers before embedding layer for user inputs;
            - ``droprate_b`` sets the dropout rate between hidden layers before embedding layer for system actions;
            - ``droprate_rnn`` sets the recurrent dropout rate on the LSTM hidden state `<https://arxiv.org/abs/1603.05118>`_;
            - ``droprate_out`` sets the dropout rate on the output of the LSTM before embedding layer for the current time step of the dialogue;

    .. note:: Droprate should be between ``0`` and ``1``, e.g. ``droprate=0.1`` would drop out ``10%`` of input units

    .. note:: For ``cosine`` similarity ``mu_pos`` and ``mu_neg`` should be between ``-1`` and ``1``.

    .. note:: There is an option to use linearly increasing batch size. The idea comes from `<https://arxiv.org/abs/1711.00489>`_.
              In order to do it pass a list to ``batch_size``, e.g. ``"batch_size": [8, 32]`` (default behaviour).
              If constant ``batch_size`` is required, pass an ``int``, e.g. ``"batch_size": 8``.

    In the config, you can specify these parameters:

    .. code-block:: yaml

        pipeline:
        - name: "intent_classifier_tensorflow_embedding"
          # nn architecture
          "num_hidden_layers_a": 0
          "hidden_layer_size_a": []
          "num_hidden_layers_b": 0
          "hidden_layer_size_b": []
          "rnn_size": 64
          "batch_size": [8, 32]
          "epochs": 1
          # embedding parameters
          "embed_dim": 20
          "mu_pos": 0.8  # should be 0.0 < ... < 1.0 for 'cosine'
          "mu_neg": -0.2  # should be -1.0 < ... < 1.0 for 'cosine'
          "similarity_type": "cosine"  # string 'cosine' or 'inner'
          "num_neg": 20
          "use_max_sim_neg": true  # flag which loss function to use
          # regularization
          "C2": 0.001
          "C_emb": 0.8
          "droprate_a": 0.0
          "droprate_b": 0.0
          "droprate_rnn": 0.1
          "droprate_out": 0.1
          # attention parameters
          "use_attention": false  # flag to use attention
          "sparse_attention": false  # flag to use sparsemax for probs
          "attn_shift_range": 5  # if None, mean dialogue length / 2
          "skip_cells": false  # flag to add gate to skip rnn time step
          # visualization of accuracy
          "calc_acc_ones_in_epochs": 50,  # small values affect performance
          "calc_acc_on_num_examples": 100  # large values affect performance

    .. note:: Parameter ``mu_neg`` is set to a negative value to mimic the original
              starspace algorithm in the case ``mu_neg = mu_pos`` and ``use_max_sim_neg = False``.
              See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.
