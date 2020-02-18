:desc: Define and train customized policy configurations to optimize your
       contextual assistant for longer contexts or unseen utterances which
       require generalization.

.. _policies:

Policies
========

.. edit-link::

.. contents::
   :local:


.. _policy_file:

Configuring Policies
^^^^^^^^^^^^^^^^^^^^

The :class:`rasa.core.policies.Policy` class decides which action to take
at every step in the conversation.

There are different policies to choose from, and you can include
multiple policies in a single :class:`rasa.core.agent.Agent`.

.. note::

    Per default a maximum of 10 next actions can be predicted
    by the agent after every user message. To update this value
    you can set the environment variable ``MAX_NUMBER_OF_PREDICTIONS``
    to the desired number of maximum predictions.


Your project's ``config.yml`` file takes a ``policies`` key
which you can use to customize the policies your assistant uses.
In the example below, the last two lines show how to use a custom
policy class and pass arguments to it.

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


Max History
-----------

One important hyperparameter for Rasa Core policies is the ``max_history``.
This controls how much dialogue history the model looks at to decide which
action to take next.

You can set the ``max_history`` by passing it to your policy's ``Featurizer``
in the policy configuration yaml file.

.. note::

    Only the ``MaxHistoryTrackerFeaturizer`` uses a max history,
    whereas the ``FullDialogueTrackerFeaturizer`` always looks at
    the full conversation history. See :ref:`featurization_conversations` for details.

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
has to be `at least` 4.

If you increase your ``max_history``, your model will become bigger and
training will take longer. If you have some information that should
affect the dialogue very far into the future, you should store it as a
slot. Slot information is always available for every featurizer.


Data Augmentation
-----------------

When you train a model, by default Rasa Core will create
longer stories by randomly gluing together
the ones in your stories files.
This is because if you have stories like:

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
subsampled during training. The augmented stories are subsampled before training
since their number can quickly become very large, and we want to limit it.
The number of sampled stories is ``augmentation_factor`` x10.
By default augmentation is set to 20, resulting in a maximum of 200 augmented stories.

``--augmentation 0`` disables all augmentation behavior.
The memoization based policies are not affected by augmentation
(independent of the ``augmentation_factor``) and will automatically
ignore all augmented stories.

Action Selection
^^^^^^^^^^^^^^^^

At every turn, each policy defined in your configuration will
predict a next action with a certain confidence level. For more information
about how each policy makes its decision, read into the policy's description below.
The bot's next action is then decided by the policy that predicts with the highest confidence.

In the case that two policies predict with equal confidence (for example, the Memoization
and Mapping Policies always predict with confidence of either 0 or 1), the priority of the
policies is considered. Rasa policies have default priorities that are set to ensure the
expected outcome in the case of a tie. They look like this, where higher numbers have higher priority:

    | 5. ``FormPolicy``
    | 4. ``FallbackPolicy`` and ``TwoStageFallbackPolicy``
    | 3. ``MemoizationPolicy`` and ``AugmentedMemoizationPolicy``
    | 2. ``MappingPolicy``
    | 1. ``EmbeddingPolicy``, ``KerasPolicy``, and ``SklearnPolicy``

This priority hierarchy ensures that, for example, if there is an intent with a mapped action, but the NLU confidence is not
above the ``nlu_threshold``, the bot will still fall back. In general, it is not recommended to have more
than one policy per priority level, and some policies on the same priority level, such as the two
fallback policies, strictly cannot be used in tandem.

If you create your own policy, use these priorities as a guide for figuring out the priority of your policy.
If your policy is a machine learning policy, it should most likely have priority 1, the same as the Rasa machine
learning policies.

.. warning::
    All policy priorities are configurable via the ``priority:`` parameter in the configuration,
    but we **do not recommend** changing them outside of specific cases such as custom policies.
    Doing so can lead to unexpected and undesired bot behavior.

.. _keras_policy:

Keras Policy
^^^^^^^^^^^^

The ``KerasPolicy`` uses a neural network implemented in
`Keras <http://keras.io>`_ to select the next action.
The default architecture is based on an LSTM, but you can override the
``KerasPolicy.model_architecture`` method to implement your own architecture.


.. literalinclude:: ../../rasa/core/policies/keras_policy.py
   :dedent: 4
   :pyobject: KerasPolicy.model_architecture

and the training is run here:

.. literalinclude:: ../../rasa/core/policies/keras_policy.py
   :dedent: 4
   :pyobject: KerasPolicy.train

You can implement the model of your choice by overriding these methods,
or initialize ``KerasPolicy`` with pre-defined ``keras model``.

In order to get reproducible training results for the same inputs you can
set the ``random_seed`` attribute of the ``KerasPolicy`` to any integer.


.. _embedding_policy:

Embedding Policy
^^^^^^^^^^^^^^^^

Transformer Embedding Dialogue Policy (TEDP)

Transformer version of the Recurrent Embedding Dialogue Policy (REDP)
used in our paper: `<https://arxiv.org/abs/1811.11707>`_

This policy has a pre-defined architecture, which comprises the
following steps:

    - concatenate user input (user intent and entities),
      previous system action, slots and active form
      for each time step into an input vector
      to pre-transformer embedding layer;
    - feed it to transformer;
    - apply a dense layer to the output of the transformer
      to get embeddings of a dialogue for each time step;
    - apply a dense layer to create embeddings for system actions for each time step;
    - calculate the similarity between the
      dialogue embedding and embedded system actions.
      This step is based on the
      `StarSpace <https://arxiv.org/abs/1709.03856>`_ idea.

It is recommended to use
``state_featurizer=LabelTokenizerSingleStateFeaturizer(...)``
(see :ref:`featurization_conversations` for details).

**Configuration:**

    Configuration parameters can be passed as parameters to the
    ``EmbeddingPolicy`` within the policy configuration file.

    .. warning::

        Pass an appropriate number of ``epochs`` to the ``EmbeddingPolicy``,
        otherwise the policy will be trained only for ``1``
        epoch.

    The algorithm also has hyper-parameters to control:

        - neural network's architecture:

            - ``hidden_layers_sizes_b`` sets a list of hidden layers
              sizes before embedding layer for system actions, the number
              of hidden layers is equal to the length of the list;
            - ``transformer_size`` sets the number of units in the transfomer;
            - ``num_transformer_layers`` sets the number of transformer layers;
            - ``pos_encoding`` sets the type of positional encoding in transformer,
              it should be either ``timing`` or ``emb``;
            - ``max_seq_length`` sets maximum sequence length
              if embedding positional encodings are used;
            - ``num_heads`` sets the number of heads in multihead attention;

        - training:

            - ``batch_size`` sets the number of training examples in one
              forward/backward pass, the higher the batch size, the more
              memory space you'll need;
            - ``batch_strategy`` sets the type of batching strategy,
              it should be either ``sequence`` or ``balanced``;
            - ``epochs`` sets the number of times the algorithm will see
              training data, where one ``epoch`` equals one forward pass and
              one backward pass of all the training examples;
            - ``random_seed`` if set to any int will get reproducible
              training results for the same inputs;

        - embedding:

            - ``embed_dim`` sets the dimension of embedding space;
            - ``num_neg`` sets the number of incorrect intent labels,
              the algorithm will minimize their similarity to the user
              input during training;
            - ``similarity_type`` sets the type of the similarity,
              it should be either ``auto``, ``cosine`` or ``inner``,
              if ``auto``, it will be set depending on ``loss_type``,
              ``inner`` for ``softmax``, ``cosine`` for ``margin``;
            - ``loss_type`` sets the type of the loss function,
              it should be either ``softmax`` or ``margin``;
            - ``ranking_length`` defines the number of top confidences over
              which to normalize ranking results if ``loss_type: "softmax"``;
              to turn off normalization set it to 0
            - ``mu_pos`` controls how similar the algorithm should try
              to make embedding vectors for correct intent labels,
              used only if ``loss_type`` is set to ``margin``;
            - ``mu_neg`` controls maximum negative similarity for
              incorrect intents,
              used only if ``loss_type`` is set to ``margin``;
            - ``use_max_sim_neg`` if ``true`` the algorithm only
              minimizes maximum similarity over incorrect intent labels,
              used only if ``loss_type`` is set to ``margin``;
            - ``scale_loss`` if ``true`` the algorithm will downscale the loss
              for examples where correct label is predicted with high confidence,
              used only if ``loss_type`` is set to ``softmax``;

        - regularization:

            - ``C2`` sets the scale of L2 regularization
            - ``C_emb`` sets the scale of how important is to minimize
              the maximum similarity between embeddings of different
              intent labels, used only if ``loss_type`` is set to ``margin``;
            - ``droprate_a`` sets the dropout rate between
              layers before embedding layer for user inputs;
            - ``droprate_b`` sets the dropout rate between layers
              before embedding layer for system actions;

        - train accuracy calculation:

            - ``evaluate_every_num_epochs`` sets how often to calculate
              train accuracy, small values may hurt performance;
            - ``evaluate_on_num_examples`` how many examples to use for
              hold out validation set to calculate of validation accuracy,
              large values may hurt performance.

    .. warning::

        Default ``max_history`` for this policy is ``None`` which means it'll use
        the ``FullDialogueTrackerFeaturizer``. We recommend to set ``max_history`` to
        some finite value in order to use ``MaxHistoryTrackerFeaturizer``
        for **faster training**. See :ref:`featurization_conversations` for details.
        We recommend to increase ``batch_size`` for ``MaxHistoryTrackerFeaturizer``
        (e.g. ``"batch_size": [32, 64]``)

    .. warning::

        If ``evaluate_on_num_examples`` is non zero, random examples will be
        picked by stratified split and used as **hold out** validation set,
        so they will be excluded from training data.
        We suggest to set it to zero if data set contains a lot of unique examples
        of dialogue turns

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
       :dedent: 4
       :start-after: # default properties (DOC MARKER - don't remove)
       :end-before: # end default properties (DOC MARKER - don't remove)

    .. note::

          Parameter ``mu_neg`` is set to a negative value to mimic
          the original starspace algorithm in the case
          ``mu_neg = mu_pos`` and ``use_max_sim_neg = False``. See
          `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.

.. _mapping-policy:

Mapping Policy
^^^^^^^^^^^^^^

The ``MappingPolicy`` can be used to directly map intents to actions. The
mappings are assigned by giving an intent the property ``triggers``, e.g.:

.. code-block:: yaml

  intents:
   - ask_is_bot:
       triggers: action_is_bot

An intent can only be mapped to at most one action. The bot will run
the mapped action once it receives a message of the triggering intent. Afterwards,
it will listen for the next message. With the next
user message, normal prediction will resume.

If you do not want your intent-action mapping to affect the dialogue
history, the mapped action must return a ``UserUtteranceReverted()``
event. This will delete the user's latest message, along with any events that
happened after it, from the dialogue history. This means you should not
include the intent-action interaction in your stories.

For example, if a user asks "Are you a bot?" off-topic in the middle of the
flow, you probably want to answer without that interaction affecting the next
action prediction. A triggered custom action can do anything, but here's a
simple example that dispatches a bot utterance and then reverts the interaction:

.. code-block:: python

  class ActionIsBot(Action):
  """Revertible mapped action for utter_is_bot"""

  def name(self):
      return "action_is_bot"

  def run(self, dispatcher, tracker, domain):
      dispatcher.utter_template(template="utter_is_bot")
      return [UserUtteranceReverted()]

.. note::

  If you use the ``MappingPolicy`` to predict bot utterance actions directly (e.g.
  ``triggers: utter_{}``), these interactions must go in your stories, as in this
  case there is no ``UserUtteranceReverted()`` and the
  intent and the mapped response action will appear in the dialogue history.

.. note::

  The MappingPolicy is also responsible for executing the default actions ``action_back``
  and ``action_restart`` in response to ``/back`` and ``/restart``. If it is not included
  in your policy example these intents will not work.

Memoization Policy
^^^^^^^^^^^^^^^^^^

The ``MemoizationPolicy`` just memorizes the conversations in your
training data. It predicts the next action with confidence ``1.0``
if this exact conversation exists in the training data, otherwise it
predicts ``None`` with confidence ``0.0``.

Augmented Memoization Policy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``AugmentedMemoizationPolicy`` remembers examples from training
stories for up to ``max_history`` turns, just like the ``MemoizationPolicy``.
Additionally, it has a forgetting mechanism that will forget a certain amount
of steps in the conversation history and try to find a match in your stories
with the reduced history. It predicts the next action with confidence ``1.0``
if a match is found, otherwise it predicts ``None`` with confidence ``0.0``.

.. note::

  If you have dialogues where some slots that are set during
  prediction time might not be set in training stories (e.g. in training
  stories starting with a reminder not all previous slots are set),
  make sure to add the relevant stories without slots to your training
  data as well.

.. _fallback-policy:

Fallback Policy
^^^^^^^^^^^^^^^

The ``FallbackPolicy`` invokes a :ref:`fallback action
<fallback-actions>` if at least one of the following occurs:

1. The intent recognition has a confidence below ``nlu_threshold``.
2. The highest ranked intent differs in confidence with the second highest 
   ranked intent by less than ``ambiguity_threshold``.
3. None of the dialogue policies predict an action with confidence higher than ``core_threshold``.

**Configuration:**

    The thresholds and fallback action can be adjusted in the policy configuration
    file as parameters of the ``FallbackPolicy``:

    .. code-block:: yaml

      policies:
        - name: "FallbackPolicy"
          nlu_threshold: 0.3
          ambiguity_threshold: 0.1
          core_threshold: 0.3
          fallback_action_name: 'action_default_fallback'

    +----------------------------+---------------------------------------------+
    | ``nlu_threshold``          | Min confidence needed to accept an NLU      |
    |                            | prediction                                  |
    +----------------------------+---------------------------------------------+
    | ``ambiguity_threshold``    | Min amount by which the confidence of the   |
    |                            | top intent must exceed that of the second   |
    |                            | highest ranked intent.                      |
    +----------------------------+---------------------------------------------+
    | ``core_threshold``         | Min confidence needed to accept an action   |
    |                            | prediction from Rasa Core                   |
    +----------------------------+---------------------------------------------+
    | ``fallback_action_name``   | Name of the :ref:`fallback action           |
    |                            | <fallback-actions>`                         |
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
                                 nlu_threshold=0.3,
                                 ambiguity_threshold=0.1)

       agent = Agent("domain.yml", policies=[KerasPolicy(), fallback])

    .. note::

       You can include either the ``FallbackPolicy`` or the
       ``TwoStageFallbackPolicy`` in your configuration, but not both.


Two-Stage Fallback Policy
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``TwoStageFallbackPolicy`` handles low NLU confidence in multiple stages
by trying to disambiguate the user input.

- If an NLU prediction has a low confidence score or is not significantly higher
  than the second highest ranked prediction, the user is asked to affirm
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
            ambiguity_threshold: 0.1
            core_threshold: 0.3
            fallback_core_action_name: "action_default_fallback"
            fallback_nlu_action_name: "action_default_fallback"
            deny_suggestion_intent_name: "out_of_scope"

    +-------------------------------+------------------------------------------+
    | ``nlu_threshold``             | Min confidence needed to accept an NLU   |
    |                               | prediction                               |
    +-------------------------------+------------------------------------------+
    | ``ambiguity_threshold``       | Min amount by which the confidence of the|
    |                               | top intent must exceed that of the second|
    |                               | highest ranked intent.                   |
    +-------------------------------+------------------------------------------+
    | ``core_threshold``            | Min confidence needed to accept an action|
    |                               | prediction from Rasa Core                |
    +-------------------------------+------------------------------------------+
    | ``fallback_core_action_name`` | Name of the :ref:`fallback action        |
    |                               | <fallback-actions>`                      |
    |                               | to be called if the confidence of Rasa   |
    |                               | Core action prediction is below the      |
    |                               | ``core_threshold``. This action is       |  
    |                               | to propose the recognized intents        |
    +-------------------------------+------------------------------------------+
    | ``fallback_nlu_action_name``  | Name of the :ref:`fallback action        |
    |                               | <fallback-actions>`                      |
    |                               | to be called if the confidence of Rasa   |
    |                               | NLU intent classification is below the   |
    |                               | ``nlu_threshold``. This action is called |
    |                               | when the user denies the second time     |
    +-------------------------------+------------------------------------------+
    |``deny_suggestion_intent_name``| The name of the intent which is used to  |
    |                               | detect that the user denies the suggested|
    |                               | intents                                  |
    +-------------------------------+------------------------------------------+

    .. note::

      You can include either the ``FallbackPolicy`` or the
      ``TwoStageFallbackPolicy`` in your configuration, but not both.



Form Policy
^^^^^^^^^^^

The ``FormPolicy`` is an extension of the ``MemoizationPolicy`` which
handles the filling of forms. Once a ``FormAction`` is called, the
``FormPolicy`` will continually predict the ``FormAction`` until all required
slots in the form are filled. For more information, see :ref:`forms`.
