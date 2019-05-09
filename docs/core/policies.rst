:desc: Define and train custom policies to optimize your contextual assistant
       for longer context or unseen utterances which require generalization.

.. _policies:

Training and Policies
=====================

.. contents::


Training
--------

Rasa Core works by creating training data from your stories and
training a model on that data.

You can run training from the command line like in the :ref:`quickstart`:


.. code-block:: bash

   rasa train core -d domain.yml -s data/stories.md \
     -o models -c config.yml

Or by creating an agent and running the train method yourself:

.. testcode::

   from rasa.core.agent import Agent

   agent = Agent()
   data = agent.load_data("stories.md")
   agent.train(data)


Training Script Options
^^^^^^^^^^^^^^^^^^^^^^^

.. program-output:: rasa train core --help


Data Augmentation
^^^^^^^^^^^^^^^^^

By default, Rasa Core will create longer stories by randomly gluing together
the ones in your stories file. This is because if you have stories like:

.. code-block:: story

    # thanks
    * thankyou
       - utter_youarewelcome

    # bye
    * goodbye
       - utter_goodbye


You actually want to teach your policy to **ignore** the dialogue history
when it isn't relevant and just respond with the same action no matter
what happened before.

You can alter this behaviour with the ``--augmentation`` flag.
Which allows you to set the ``augmentation_factor``.
The ``augmentation_factor`` determines how many augmented stories are
subsampled during training. Subsampling of the augmented stories is done in order to
not get too many stories from augmentation, since their number
can become very large quickly.
The number of sampled stories is ``augmentation_factor`` x10.

``--augmentation 0`` disables all augmentation behavior.
The memoization based policies are not affected by augmentation
(independent of the ``augmentation_factor``) and will automatically
ignore all augmented stories.

In python, you can pass the ``augmentation_factor`` argument to the
``Agent.load_data`` method.
By default augmentation is set to 20, resulting in a maximum of 200 augmented stories.

Policies
--------

The :class:`rasa.core.policies.Policy` class decides which action to take
at every step in the conversation.

There are different policies to choose from, and you can include
multiple policies in a single :class:`rasa.core.agent.Agent`. At
every turn, the policy which predicts the next action with the
highest confidence will be used. If two policies predict with equal
confidence, the policy with the higher priority will be used.

Configuration of Policies
^^^^^^^^^^^^^^^^^^^^^^^^^

.. _policy_file:

Configuring policies using a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using the training script, you must set the policies you would like
the Core model to use in a YAML file.

For example:

.. code-block:: yaml

  policies:
    - name: "KerasPolicy"
      featurizer:
      - name: MaxHistoryTrackerFeaturizer
        max_history: 5
        state_featurizer:
          - name: BinarySingleStateFeaturizer
    - name: "MemoizationPolicy"
      max_history: 5
    - name: "FallbackPolicy"
      nlu_threshold: 0.4
      core_threshold: 0.3
      fallback_action_name: "my_fallback_action"
    - name: "path.to.your.policy.class"
      arg1: "..."

Pass the YAML file's name to the train script using the ``--config``
argument (or just ``-c``). There is a default config file you can use to
get started in ``default_config.yml``, which is already implemented
as the provided ``policies.yml`` file in the starter pack.


.. _default_config:

Default configuration
~~~~~~~~~~~~~~~~~~~~~

By default, we try to provide you with a good set of configuration values
and policies that suit most people. But you are encouraged to modify
these to your needs:

.. literalinclude:: ../../rasa/cli/default_config.yml


Max History
~~~~~~~~~~~

One important hyperparameter for Rasa Core policies is the ``max_history``.
This controls how much dialogue history the model looks at to decide which
action to take next.

You can set the ``max_history`` by passing it to your policy's ``Featurizer``
in the policy configuration yaml file.

.. note::

    Only the ``MaxHistoryTrackerFeaturizer`` uses a max history,
    whereas the ``FullDialogueTrackerFeaturizer`` always looks at
    the full conversation history.

As an example, let's say you have an ``out_of_scope`` intent which
describes off-topic user messages. If your bot sees this intent multiple
times in a row, you might want to tell the user what you `can` help them
with. So your story might look like this:

.. code-block:: story

   * out_of_scope
      - utter_default
   * out_of_scope
      - utter_default
   * out_of_scope
      - utter_help_message

For Rasa Core to learn this pattern, the ``max_history``
has to be `at least` ``3``.

If you increase your ``max_history``, your model will become bigger and
training will take longer. If you have some information that should
affect the dialogue very far into the future, you should store it as a
slot. Slot information is always available for every featurizer.


Configuring policies in code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can load a policy configuration file named ``"policies.yml"`` from your
code like so:

.. code-block:: python

    from rasa.core import config as policy_config
    from rasa.core.agent import Agent

    policies = policy_config.load("policies.yml")
    agent = Agent("domain.yml", policies=policies)


Alternatively, you can pass a list of policies directly
when you create an agent:

.. code-block:: python

   from rasa.core.policies.memoization import MemoizationPolicy
   from rasa.core.policies.keras_policy import KerasPolicy
   from rasa.core.agent import Agent

   agent = Agent("domain.yml",
                 policies=[MemoizationPolicy(), KerasPolicy()])

Keras Policy
^^^^^^^^^^^^

The ``KerasPolicy`` uses a neural network implemented in
`Keras <http://keras.io>`_ to select the next action.
The default architecture is based on an LSTM, but you can override the
``KerasPolicy.model_architecture`` method to implement your own architecture.


.. literalinclude:: ../../rasa/core/policies/keras_policy.py
   :pyobject: KerasPolicy.model_architecture

and the training is run here:

.. literalinclude:: ../../rasa/core/policies/keras_policy.py
   :pyobject: KerasPolicy.train

You can implement the model of your choice by overriding these methods,
or initialize ``KerasPolicy`` with pre-defined ``keras model``.

In order to get reproducible training results for the same inputs you can
set the ``random_seed`` attribute of the ``KerasPolicy`` to any integer.


.. _embedding_policy:

Embedding Policy
^^^^^^^^^^^^^^^^

The Recurrent Embedding Dialogue Policy (REDP)
described in our paper: `<https://arxiv.org/abs/1811.11707>`_

This policy has a pre-defined architecture, which comprises the
following steps:

    - apply dense layers to create embeddings for user intents,
      entities and system actions including previous actions and slots;
    - use the embeddings of previous user inputs as a user memory
      and embeddings of previous system actions as a system memory;
    - concatenate user input, previous system action and slots
      embeddings for current time into an input vector to rnn;
    - using user and previous system action embeddings from the input
      vector, calculate attention probabilities over the user and
      system memories (for system memory, this policy uses
      `NTM mechanism <https://arxiv.org/abs/1410.5401>`_ with attention
      by location);
    - sum the user embedding and user attention vector and feed it
      and the embeddings of the slots as an input to an LSTM cell;
    - apply a dense layer to the output of the LSTM to get a raw
      recurrent embedding of a dialogue;
    - sum this raw recurrent embedding of a dialogue with system
      attention vector to create dialogue level embedding, this step
      allows the algorithm to repeat previous system action by copying
      its embedding vector directly to the current time output;
    - weight previous LSTM states with system attention probabilities
      to get the previous action embedding, the policy is likely payed
      attention to;
    - if the similarity between this previous action embedding and
      current time dialogue embedding is high, overwrite current LSTM
      state with the one from the time when this action happened;
    - for each LSTM time step, calculate the similarity between the
      dialogue embedding and embedded system actions.
      This step is based on the
      `starspace idea <https://arxiv.org/abs/1709.03856>`_.

.. note::

    This policy only works with
    ``FullDialogueTrackerFeaturizer(state_featurizer)``.

It is recommended to use
``state_featurizer=LabelTokenizerSingleStateFeaturizer(...)``
(see :ref:`featurization` for details).

**Configuration:**

    Configuration parameters can be passed as parameters to the
    ``EmbeddingPolicy`` within the policy configuration file.

    .. note::

        Pass an appropriate number of ``epochs`` to the ``EmbeddingPolicy``,
        otherwise the policy will be trained only for ``1``
        epoch. Since this is an embedding based policy, it requires a large
        number of epochs, which depends on the complexity of the
        training data and whether attention is used or not.

    The main feature of this policy is an **attention** mechanism over
    previous user input and system actions.
    **Attention is turned on by default**; in order to turn it off,
    configure the following parameters:

        - ``attn_before_rnn`` if ``true`` the algorithm will use
          attention mechanism over previous user input, default ``true``;
        - ``attn_after_rnn`` if ``true`` the algorithm will use
          attention mechanism over previous system actions and will be
          able to copy previously executed action together with LSTM's
          hidden state from its history, default ``true``;
        - ``sparse_attention`` if ``true`` ``sparsemax`` will be used
          instead of ``softmax`` for attention probabilities, default
          ``false``;
        - ``attn_shift_range`` the range of allowed location-based
          attention shifts for system memory (``attn_after_rnn``), see
          `<https://arxiv.org/abs/1410.5401>`_ for details;

    .. note::

        Attention requires larger values of ``epochs`` and takes longer
        to train. But it can learn more complicated and nonlinear behaviour.

    The algorithm also has hyper-parameters to control:

        - neural network's architecture:

            - ``hidden_layers_sizes_a`` sets a list of hidden layers
              sizes before embedding layer for user inputs, the number
              of hidden layers is equal to the length of the list;
            - ``hidden_layers_sizes_b`` sets a list of hidden layers
              sizes before embedding layer for system actions, the number
              of hidden layers is equal to the length of the list;
            - ``rnn_size`` sets the number of units in the LSTM cell;

        - training:

            - ``layer_norm`` if ``true`` layer normalization for lstm
              cell is turned on,  default ``true``;
            - ``batch_size`` sets the number of training examples in one
              forward/backward pass, the higher the batch size, the more
              memory space you'll need;
            - ``epochs`` sets the number of times the algorithm will see
              training data, where one ``epoch`` equals one forward pass and
              one backward pass of all the training examples;
            - ``random_seed`` if set to any int will get reproducible
              training results for the same inputs;

        - embedding:

            - ``embed_dim`` sets the dimension of embedding space;
            - ``mu_pos`` controls how similar the algorithm should try
              to make embedding vectors for correct intent labels;
            - ``mu_neg`` controls maximum negative similarity for
              incorrect intents;
            - ``similarity_type`` sets the type of the similarity,
              it should be either ``cosine`` or ``inner``;
            - ``num_neg`` sets the number of incorrect intent labels,
              the algorithm will minimize their similarity to the user
              input during training;
            - ``use_max_sim_neg`` if ``true`` the algorithm only
              minimizes maximum similarity over incorrect intent labels;

        - regularization:

            - ``C2`` sets the scale of L2 regularization
            - ``C_emb`` sets the scale of how important is to minimize
              the maximum similarity between embeddings of different
              intent labels;
            - ``droprate_a`` sets the dropout rate between hidden
              layers before embedding layer for user inputs;
            - ``droprate_b`` sets the dropout rate between hidden layers
              before embedding layer for system actions;
            - ``droprate_rnn`` sets the recurrent dropout rate on
              the LSTM hidden state `<https://arxiv.org/abs/1603.05118>`_;

        - train accuracy calculation:

            - ``evaluate_every_num_epochs`` sets how often to calculate
              train accuracy, small values may hurt performance;
            - ``evaluate_on_num_examples`` how many examples to use for
              calculation of train accuracy, large values may hurt
              performance.

    .. note::

        Droprate should be between ``0`` and ``1``, e.g.
        ``droprate=0.1`` would drop out ``10%`` of input units.

    .. note::

        For ``cosine`` similarity ``mu_pos`` and ``mu_neg`` should
        be between ``-1`` and ``1``.

    .. note::

        There is an option to use linearly increasing batch size.
        The idea comes from `<https://arxiv.org/abs/1711.00489>`_.
        In order to do it pass a list to ``batch_size``, e.g.
        ``"batch_size": [8, 32]`` (default behaviour). If constant
        ``batch_size`` is required, pass an ``int``, e.g.
        ``"batch_size": 8``.

    These parameters can be specified in the policy configuration file.
    The default values are defined in ``EmbeddingPolicy.defaults``:

    .. literalinclude:: ../../rasa/core/policies/embedding_policy.py
       :start-after: # default properties (DOC MARKER - don't remove)
       :end-before: # end default properties (DOC MARKER - don't remove)

    .. note::

          Parameter ``mu_neg`` is set to a negative value to mimic
          the original starspace algorithm in the case
          ``mu_neg = mu_pos`` and ``use_max_sim_neg = False``. See
          `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.

.. _memoization_policy:

Memoization Policy
^^^^^^^^^^^^^^^^^^

The ``MemoizationPolicy`` just memorizes the conversations in your
training data. It predicts the next action with confidence ``1.0``
if this exact conversation exists in the training data, otherwise it
predicts ``None`` with confidence ``0.0``.


Form Policy
^^^^^^^^^^^

The ``FormPolicy`` is an extension of the ``MemoizationPolicy`` which
handles the filling of forms. Once a ``FormAction`` is called, the
``FormPolicy`` will continually predict the ``FormAction`` until all slots
in the form are filled. For more information, see `Slot Filling
<https://rasa.com/docs/core/slotfilling/>`_.


Mapping Policy
^^^^^^^^^^^^^^

The ``MappingPolicy`` can be used to directly map intents to actions such that
the mapped action will always be executed. The mappings are assigned by giving
and intent the property `'triggers'`, e.g.:

.. code-block:: yaml

  intents:
   - greet: {triggers: utter_goodbye}

An intent can only be mapped to at most one action. The bot will run
the action once it receives a message of the mapped intent. Afterwards,
it will listen for the next message.

.. note::

  The mapping policy will predict the mapped action after the intent (e.g.
  ``utter_goodbye`` in the above example) and afterwards it will wait for
  the next user message (predicting ``action_listen``). With the next
  user message normal prediction will resume.

  You should have an example like

  .. code-block:: story

    * greet
      - utter_goodbye

  in your stories. Otherwise any machine learning policy might be confused
  by the sudden appearance of the predicted ``action_greet`` in
  the dialouge history.

.. _fallback_policy:

Fallback Policy
^^^^^^^^^^^^^^^

The ``FallbackPolicy`` invokes a `fallback action
<https://rasa.com/docs/core/fallbacks/>`_ if the intent recognition
has a confidence below ``nlu_threshold`` or if none of the dialogue
policies predict an action with confidence higher than ``core_threshold``.

**Configuration:**

    The thresholds and fallback action can be adjusted in the policy configuration
    file as parameters of the ``FallbackPolicy``:

    .. code-block:: yaml

      policies:
        - name: "FallbackPolicy"
          nlu_threshold: 0.3
          core_threshold: 0.3
          fallback_action_name: 'action_default_fallback'

    +----------------------------+---------------------------------------------+
    | ``nlu_threshold``          | Min confidence needed to accept an NLU      |
    |                            | prediction                                  |
    +----------------------------+---------------------------------------------+
    | ``core_threshold``         | Min confidence needed to accept an action   |
    |                            | prediction from Rasa Core                   |
    +----------------------------+---------------------------------------------+
    | ``fallback_action_name``   | Name of the `fallback action                |
    |                            | <https://rasa.com/docs/core/fallbacks/>`_   |
    |                            | to be called if the confidence of intent    |
    |                            | or action is below the respective threshold |
    +----------------------------+---------------------------------------------+

    You can also configure the ``FallbackPolicy`` in your python code:

    .. code-block:: python

       from rasa.core.policies.fallback import FallbackPolicy
       from rasa.core.policies.keras_policy import KerasPolicy
       from rasa.core.agent import Agent

       fallback = FallbackPolicy(fallback_action_name="action_default_fallback",
                                 core_threshold=0.3,
                                 nlu_threshold=0.3)

       agent = Agent("domain.yml", policies=[KerasPolicy(), fallback])

    .. note::

       You can include either the ``FallbackPolicy`` or the
       ``TwoStageFallbackPolicy`` in your configuration, but not both.


Two-Stage Fallback Policy
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``TwoStageFallbackPolicy`` handles low NLU confidence in multiple stages
by trying to disambiguate the user input.

- If a NLU prediction has a low confidence score, the user is asked to affirm
  the classification of the intent.

    - If they affirm, the story continues as if the intent was classified
      with high confidence from the beginning.
    - If they deny, the user is asked to rephrase their message.

- Rephrasing

    - If the classification of the rephrased intent was confident, the story
      continues as if the user had this intent from the beginning.
    - If the rephrased intent was not classified with high confidence, the user
      is asked to affirm the classified intent.

- Second affirmation

    - If the user affirms the intent, the story continues as if the user had
      this intent from the beginning.
    - If the user denies, the original intent is classified as the specified
      ``deny_suggestion_intent_name``, and an ultimate fallback action
      is triggered (e.g. a handoff to a human).

**Configuration:**

    To use the ``TwoStageFallbackPolicy``, include the following in your
    policy configuration.

    .. code-block:: yaml

        policies:
          - name: TwoStageFallbackPolicy
            nlu_threshold: 0.3
            core_threshold: 0.3
            fallback_core_action_name: "action_default_fallback"
            fallback_nlu_action_name: "action_default_fallback"
            deny_suggestion_intent_name: "out_of_scope"

    +-------------------------------+------------------------------------------+
    | ``nlu_threshold``             | Min confidence needed to accept an NLU   |
    |                               | prediction                               |
    +-------------------------------+------------------------------------------+
    | ``core_threshold``            | Min confidence needed to accept an action|
    |                               | prediction from Rasa Core                |
    +-------------------------------+------------------------------------------+
    | ``fallback_core_action_name`` | Name of the `fallback action             |
    |                               | <https://rasa.com/docs/core/fallbacks/>`_|
    |                               | to be called if the confidence of Rasa   |
    |                               | Core action prediction is below the      |
    |                               | ``core_threshold``                       |
    +-------------------------------+------------------------------------------+
    | ``fallback_nlu_action_name``  | Name of the `fallback action             |
    |                               | <https://rasa.com/docs/core/fallbacks/>`_|
    |                               | to be called if the confidence of Rasa   |
    |                               | NLU intent classification is below the   |
    |                               | ``nlu_threshold``                        |
    +-------------------------------+------------------------------------------+
    |``deny_suggestion_intent_name``| The name of the intent which is used to  |
    |                               | detect that the user denies the suggested|
    |                               | intents                                  |
    +-------------------------------+------------------------------------------+

    .. note::

      You can include either the ``FallbackPolicy`` or the
      ``TwoStageFallbackPolicy`` in your configuration, but not both.

.. include:: feedback.inc
