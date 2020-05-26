:desc: Define and train customized policy configurations to optimize your a
       contextual assistant for longer contexts or unseen utterances which a
       require generalization. a
 a
.. _policies: a
 a
Policies a
======== a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
 a
.. _policy_file: a
 a
Configuring Policies a
^^^^^^^^^^^^^^^^^^^^ a
 a
The :class:`rasa.core.policies.Policy` class decides which action to take a
at every step in the conversation. a
 a
There are different policies to choose from, and you can include a
multiple policies in a single :class:`rasa.core.agent.Agent`. a
 a
.. note:: a
 a
    Per default a maximum of 10 next actions can be predicted a
    by the agent after every user message. To update this value a
    you can set the environment variable ``MAX_NUMBER_OF_PREDICTIONS`` a
    to the desired number of maximum predictions. a
 a
 a
Your project's ``config.yml`` file takes a ``policies`` key a
which you can use to customize the policies your assistant uses. a
In the example below, the last two lines show how to use a custom a
policy class and pass arguments to it. a
 a
.. code-block:: yaml a
 a
  policies: a
    - name: "TEDPolicy" a
      featurizer: a
      - name: MaxHistoryTrackerFeaturizer a
        max_history: 5 a
        state_featurizer: a
          - name: BinarySingleStateFeaturizer a
    - name: "MemoizationPolicy" a
      max_history: 5 a
    - name: "FallbackPolicy" a
      nlu_threshold: 0.4 a
      core_threshold: 0.3 a
      fallback_action_name: "my_fallback_action" a
    - name: "path.to.your.policy.class" a
      arg1: "..." a
 a
 a
Max History a
----------- a
 a
One important hyperparameter for Rasa Core policies is the ``max_history``. a
This controls how much dialogue history the model looks at to decide which a
action to take next. a
 a
You can set the ``max_history`` by passing it to your policy's ``Featurizer`` a
in the policy configuration yaml file. a
 a
.. note:: a
 a
    Only the ``MaxHistoryTrackerFeaturizer`` uses a max history, a
    whereas the ``FullDialogueTrackerFeaturizer`` always looks at a
    the full conversation history. See :ref:`featurization_conversations` for details. a
 a
As an example, let's say you have an ``out_of_scope`` intent which a
describes off-topic user messages. If your bot sees this intent multiple a
times in a row, you might want to tell the user what you `can` help them a
with. So your story might look like this: a
 a
.. code-block:: story a
 a
   * out_of_scope a
      - utter_default a
   * out_of_scope a
      - utter_default a
   * out_of_scope a
      - utter_help_message a
 a
For Rasa Core to learn this pattern, the ``max_history`` a
has to be `at least` 4. a
 a
If you increase your ``max_history``, your model will become bigger and a
training will take longer. If you have some information that should a
affect the dialogue very far into the future, you should store it as a a
slot. Slot information is always available for every featurizer. a
 a
 a
Data Augmentation a
----------------- a
 a
When you train a model, by default Rasa Core will create a
longer stories by randomly gluing together a
the ones in your stories files. a
This is because if you have stories like: a
 a
.. code-block:: story a
 a
    # thanks a
    * thankyou a
       - utter_youarewelcome a
 a
    # bye a
    * goodbye a
       - utter_goodbye a
 a
 a
You actually want to teach your policy to **ignore** the dialogue history a
when it isn't relevant and just respond with the same action no matter a
what happened before. a
 a
You can alter this behaviour with the ``--augmentation`` flag. a
Which allows you to set the ``augmentation_factor``. a
The ``augmentation_factor`` determines how many augmented stories are a
subsampled during training. The augmented stories are subsampled before training a
since their number can quickly become very large, and we want to limit it. a
The number of sampled stories is ``augmentation_factor`` x10. a
By default augmentation is set to 20, resulting in a maximum of 200 augmented stories. a
 a
``--augmentation 0`` disables all augmentation behavior. a
The memoization based policies are not affected by augmentation a
(independent of the ``augmentation_factor``) and will automatically a
ignore all augmented stories. a
 a
Action Selection a
^^^^^^^^^^^^^^^^ a
 a
At every turn, each policy defined in your configuration will a
predict a next action with a certain confidence level. For more information a
about how each policy makes its decision, read into the policy's description below. a
The bot's next action is then decided by the policy that predicts with the highest confidence. a
 a
In the case that two policies predict with equal confidence (for example, the Memoization a
and Mapping Policies always predict with confidence of either 0 or 1), the priority of the a
policies is considered. Rasa policies have default priorities that are set to ensure the a
expected outcome in the case of a tie. They look like this, where higher numbers have higher priority: a
 a
    | 5. ``FormPolicy`` a
    | 4. ``FallbackPolicy`` and ``TwoStageFallbackPolicy`` a
    | 3. ``MemoizationPolicy`` and ``AugmentedMemoizationPolicy`` a
    | 2. ``MappingPolicy`` a
    | 1. ``TEDPolicy`` and ``SklearnPolicy`` a
 a
This priority hierarchy ensures that, for example, if there is an intent with a mapped action, but the NLU confidence is not a
above the ``nlu_threshold``, the bot will still fall back. In general, it is not recommended to have more a
than one policy per priority level, and some policies on the same priority level, such as the two a
fallback policies, strictly cannot be used in tandem. a
 a
If you create your own policy, use these priorities as a guide for figuring out the priority of your policy. a
If your policy is a machine learning policy, it should most likely have priority 1, the same as the Rasa machine a
learning policies. a
 a
.. warning:: a
    All policy priorities are configurable via the ``priority:`` parameter in the configuration, a
    but we **do not recommend** changing them outside of specific cases such as custom policies. a
    Doing so can lead to unexpected and undesired bot behavior. a
 a
.. _embedding_policy: a
 a
Embedding Policy a
^^^^^^^^^^^^^^^^ a
 a
    .. warning:: a
 a
        ``EmbeddingPolicy`` was renamed to ``TEDPolicy``. Please use :ref:`ted_policy` instead of ``EmbeddingPolicy`` a
        in your policy configuration. The functionality of the policy stayed the same. a
 a
.. _ted_policy: a
 a
TED Policy a
^^^^^^^^^^ a
 a
The Transformer Embedding Dialogue (TED) Policy is described in a
`our paper <https://arxiv.org/abs/1910.00486>`__. a
 a
This policy has a pre-defined architecture, which comprises the a
following steps: a
 a
    - concatenate user input (user intent and entities), previous system actions, slots and active forms for each time a
      step into an input vector to pre-transformer embedding layer; a
    - feed it to transformer; a
    - apply a dense layer to the output of the transformer to get embeddings of a dialogue for each time step; a
    - apply a dense layer to create embeddings for system actions for each time step; a
    - calculate the similarity between the dialogue embedding and embedded system actions. a
      This step is based on the `StarSpace <https://arxiv.org/abs/1709.03856>`_ idea. a
 a
It is recommended to use ``state_featurizer=LabelTokenizerSingleStateFeaturizer(...)`` a
(see :ref:`featurization_conversations` for details). a
 a
**Configuration:** a
 a
    Configuration parameters can be passed as parameters to the ``TEDPolicy`` within the configuration file. a
    If you want to adapt your model, start by modifying the following parameters: a
 a
        - ``epochs``: a
          This parameter sets the number of times the algorithm will see the training data (default: ``1``). a
          One ``epoch`` is equals to one forward pass and one backward pass of all the training examples. a
          Sometimes the model needs more epochs to properly learn. a
          Sometimes more epochs don't influence the performance. a
          The lower the number of epochs the faster the model is trained. a
        - ``hidden_layers_sizes``: a
          This parameter allows you to define the number of feed forward layers and their output a
          dimensions for dialogues and intents (default: ``dialogue: [], label: []``). a
          Every entry in the list corresponds to a feed forward layer. a
          For example, if you set ``dialogue: [256, 128]``, we will add two feed forward layers in front of a
          the transformer. The vectors of the input tokens (coming from the dialogue) will be passed on to those a
          layers. The first layer will have an output dimension of 256 and the second layer will have an output a
          dimension of 128. If an empty list is used (default behaviour), no feed forward layer will be a
          added. a
          Make sure to use only positive integer values. Usually, numbers of power of two are used. a
          Also, it is usual practice to have decreasing values in the list: next value is smaller or equal to the a
          value before. a
        - ``number_of_transformer_layers``: a
          This parameter sets the number of transformer layers to use (default: ``1``). a
          The number of transformer layers corresponds to the transformer blocks to use for the model. a
        - ``transformer_size``: a
          This parameter sets the number of units in the transformer (default: ``128``). a
          The vectors coming out of the transformers will have the given ``transformer_size``. a
        - ``weight_sparsity``: a
          This parameter defines the fraction of kernel weights that are set to 0 for all feed forward layers a
          in the model (default: ``0.8``). The value should be between 0 and 1. If you set ``weight_sparsity`` a
          to 0, no kernel weights will be set to 0, the layer acts as a standard feed forward layer. You should not a
          set ``weight_sparsity`` to 1 as this would result in all kernel weights being 0, i.e. the model is not able a
          to learn. a
 a
    .. warning:: a
 a
        Pass an appropriate number, for example 50,  of ``epochs`` to the ``TEDPolicy``, otherwise the policy will a
        be trained only for ``1`` epoch. a
 a
    .. warning:: a
 a
        Default ``max_history`` for this policy is ``None`` which means it'll use the a
        ``FullDialogueTrackerFeaturizer``. We recommend to set ``max_history`` to some finite value in order to a
        use ``MaxHistoryTrackerFeaturizer`` for **faster training**. See :ref:`featurization_conversations` for a
        details. We recommend to increase ``batch_size`` for ``MaxHistoryTrackerFeaturizer`` a
        (e.g. ``"batch_size": [32, 64]``) a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            .. container:: block a
 a
                The above configuration parameters are the ones you should configure to fit your model to your data. a
                However, additional parameters exist that can be adapted. a
 a
        .. code-block:: none a
 a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | Parameter                       | Default Value    | Description                                                  | a
         +=================================+==================+==============================================================+ a
         | hidden_layers_sizes             | dialogue: []     | Hidden layer sizes for layers before the embedding layers    | a
         |                                 | label: []        | for dialogue and labels. The number of hidden layers is      | a
         |                                 |                  | equal to the length of the corresponding.                    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | transformer_size                | 128              | Number of units in transformer.                              | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | number_of_transformer_layers    | 1                | Number of transformer layers.                                | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | number_of_attention_heads       | 4                | Number of attention heads in transformer.                    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_key_relative_attention      | False            | If 'True' use key relative embeddings in attention.          | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_value_relative_attention    | False            | If 'True' use value relative embeddings in attention.        | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | max_relative_position           | None             | Maximum position for relative embeddings.                    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | batch_size                      | [8, 32]          | Initial and final value for batch sizes.                     | a
         |                                 |                  | Batch size will be linearly increased for each epoch.        | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | batch_strategy                  | "balanced"       | Strategy used when creating batches.                         | a
         |                                 |                  | Can be either 'sequence' or 'balanced'.                      | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | epochs                          | 1                | Number of epochs to train.                                   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | random_seed                     | None             | Set random seed to any 'int' to get reproducible results.    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | embedding_dimension             | 20               | Dimension size of embedding vectors.                         | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | number_of_negative_examples     | 20               | The number of incorrect labels. The algorithm will minimize  | a
         |                                 |                  | their similarity to the user input during training.          | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | similarity_type                 | "auto"           | Type of similarity measure to use, either 'auto' or 'cosine' | a
         |                                 |                  | or 'inner'.                                                  | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | loss_type                       | "softmax"        | The type of the loss function, either 'softmax' or 'margin'. | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | ranking_length                  | 10               | Number of top actions to normalize scores for loss type      | a
         |                                 |                  | 'softmax'. Set to 0 to turn off normalization.               | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | maximum_positive_similarity     | 0.8              | Indicates how similar the algorithm should try to make       | a
         |                                 |                  | embedding vectors for correct labels.                        | a
         |                                 |                  | Should be 0.0 < ... < 1.0 for 'cosine' similarity type.      | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | maximum_negative_similarity     | -0.2             | Maximum negative similarity for incorrect labels.            | a
         |                                 |                  | Should be -1.0 < ... < 1.0 for 'cosine' similarity type.     | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_maximum_negative_similarity | True             | If 'True' the algorithm only minimizes maximum similarity    | a
         |                                 |                  | over incorrect intent labels, used only if 'loss_type' is    | a
         |                                 |                  | set to 'margin'.                                             | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | scale_loss                      | True             | Scale loss inverse proportionally to confidence of correct   | a
         |                                 |                  | prediction.                                                  | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | regularization_constant         | 0.001            | The scale of regularization.                                 | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | negative_margin_scale           | 0.8              | The scale of how important it is to minimize the maximum     | a
         |                                 |                  | similarity between embeddings of different labels.           | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | drop_rate_dialogue              | 0.1              | Dropout rate for embedding layers of dialogue features.      | a
         |                                 |                  | Value should be between 0 and 1.                             | a
         |                                 |                  | The higher the value the higher the regularization effect.   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | drop_rate_label                 | 0.0              | Dropout rate for embedding layers of label features.         | a
         |                                 |                  | Value should be between 0 and 1.                             | a
         |                                 |                  | The higher the value the higher the regularization effect.   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | drop_rate_attention             | 0.0              | Dropout rate for attention. Value should be between 0 and 1. | a
         |                                 |                  | The higher the value the higher the regularization effect.   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | weight_sparsity                 | 0.8              | Sparsity of the weights in dense layers.                     | a
         |                                 |                  | Value should be between 0 and 1.                             | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | evaluate_every_number_of_epochs | 20               | How often to calculate validation accuracy.                  | a
         |                                 |                  | Set to '-1' to evaluate just once at the end of training.    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | evaluate_on_number_of_examples  | 0                | How many examples to use for hold out validation set.        | a
         |                                 |                  | Large values may hurt performance, e.g. model accuracy.      | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | tensorboard_log_directory       | None             | If you want to use tensorboard to visualize training         | a
         |                                 |                  | metrics, set this option to a valid output directory. You    | a
         |                                 |                  | can view the training metrics after training in tensorboard  | a
         |                                 |                  | via 'tensorboard --logdir <path-to-given-directory>'.        | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | tensorboard_log_level           | "epoch"          | Define when training metrics for tensorboard should be       | a
         |                                 |                  | logged. Either after every epoch ('epoch') or for every      | a
         |                                 |                  | training step ('minibatch').                                 | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
 a
        .. warning:: a
 a
            If ``evaluate_on_number_of_examples`` is non zero, random examples will be picked by stratified split and a
            used as **hold out** validation set, so they will be excluded from training data. a
            We suggest to set it to zero if data set contains a lot of unique examples of dialogue turns. a
 a
        .. note:: a
 a
            For ``cosine`` similarity ``maximum_positive_similarity`` and ``maximum_negative_similarity`` should a
            be between ``-1`` and ``1``. a
 a
        .. note:: a
 a
            There is an option to use linearly increasing batch size. The idea comes from a
            `<https://arxiv.org/abs/1711.00489>`_. In order to do it pass a list to ``batch_size``, e.g. a
            ``"batch_size": [8, 32]`` (default behaviour). If constant ``batch_size`` is required, pass an ``int``, a
            e.g. ``"batch_size": 8``. a
 a
        .. note:: a
 a
            The parameter ``maximum_negative_similarity`` is set to a negative value to mimic the original a
            starspace algorithm in the case ``maximum_negative_similarity = maximum_positive_similarity`` and a
            ``use_maximum_negative_similarity = False``. See `starspace paper <https://arxiv.org/abs/1709.03856>`_ a
            for details. a
 a
 a
.. _mapping-policy: a
 a
Mapping Policy a
^^^^^^^^^^^^^^ a
 a
The ``MappingPolicy`` can be used to directly map intents to actions. The a
mappings are assigned by giving an intent the property ``triggers``, e.g.: a
 a
.. code-block:: yaml a
 a
  intents: a
   - ask_is_bot: a
       triggers: action_is_bot a
 a
An intent can only be mapped to at most one action. The bot will run a
the mapped action once it receives a message of the triggering intent. Afterwards, a
it will listen for the next message. With the next a
user message, normal prediction will resume. a
 a
If you do not want your intent-action mapping to affect the dialogue a
history, the mapped action must return a ``UserUtteranceReverted()`` a
event. This will delete the user's latest message, along with any events that a
happened after it, from the dialogue history. This means you should not a
include the intent-action interaction in your stories. a
 a
For example, if a user asks "Are you a bot?" off-topic in the middle of the a
flow, you probably want to answer without that interaction affecting the next a
action prediction. A triggered custom action can do anything, but here's a a
simple example that dispatches a bot utterance and then reverts the interaction: a
 a
.. code-block:: python a
 a
  class ActionIsBot(Action): a
  """Revertible mapped action for utter_is_bot""" a
 a
  def name(self): a
      return "action_is_bot" a
 a
  def run(self, dispatcher, tracker, domain): a
      dispatcher.utter_template(template="utter_is_bot") a
      return [UserUtteranceReverted()] a
 a
.. note:: a
 a
  If you use the ``MappingPolicy`` to predict bot utterance actions directly (e.g. a
  ``triggers: utter_{}``), these interactions must go in your stories, as in this a
  case there is no ``UserUtteranceReverted()`` and the a
  intent and the mapped response action will appear in the dialogue history. a
 a
.. note:: a
 a
  The MappingPolicy is also responsible for executing the default actions ``action_back`` a
  and ``action_restart`` in response to ``/back`` and ``/restart``. If it is not included a
  in your policy example these intents will not work. a
 a
Memoization Policy a
^^^^^^^^^^^^^^^^^^ a
 a
The ``MemoizationPolicy`` just memorizes the conversations in your a
training data. It predicts the next action with confidence ``1.0`` a
if this exact conversation exists in the training data, otherwise it a
predicts ``None`` with confidence ``0.0``. a
 a
Augmented Memoization Policy a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The ``AugmentedMemoizationPolicy`` remembers examples from training a
stories for up to ``max_history`` turns, just like the ``MemoizationPolicy``. a
Additionally, it has a forgetting mechanism that will forget a certain amount a
of steps in the conversation history and try to find a match in your stories a
with the reduced history. It predicts the next action with confidence ``1.0`` a
if a match is found, otherwise it predicts ``None`` with confidence ``0.0``. a
 a
.. note:: a
 a
  If you have dialogues where some slots that are set during a
  prediction time might not be set in training stories (e.g. in training a
  stories starting with a reminder not all previous slots are set), a
  make sure to add the relevant stories without slots to your training a
  data as well. a
 a
.. _fallback-policy: a
 a
Fallback Policy a
^^^^^^^^^^^^^^^ a
 a
The ``FallbackPolicy`` invokes a :ref:`fallback action a
<fallback-actions>` if at least one of the following occurs: a
 a
1. The intent recognition has a confidence below ``nlu_threshold``. a
2. The highest ranked intent differs in confidence with the second highest  a
   ranked intent by less than ``ambiguity_threshold``. a
3. None of the dialogue policies predict an action with confidence higher than ``core_threshold``. a
 a
**Configuration:** a
 a
    The thresholds and fallback action can be adjusted in the policy configuration a
    file as parameters of the ``FallbackPolicy``: a
 a
    .. code-block:: yaml a
 a
      policies: a
        - name: "FallbackPolicy" a
          nlu_threshold: 0.3 a
          ambiguity_threshold: 0.1 a
          core_threshold: 0.3 a
          fallback_action_name: 'action_default_fallback' a
 a
    +----------------------------+---------------------------------------------+ a
    | ``nlu_threshold``          | Min confidence needed to accept an NLU      | a
    |                            | prediction                                  | a
    +----------------------------+---------------------------------------------+ a
    | ``ambiguity_threshold``    | Min amount by which the confidence of the   | a
    |                            | top intent must exceed that of the second   | a
    |                            | highest ranked intent.                      | a
    +----------------------------+---------------------------------------------+ a
    | ``core_threshold``         | Min confidence needed to accept an action   | a
    |                            | prediction from Rasa Core                   | a
    +----------------------------+---------------------------------------------+ a
    | ``fallback_action_name``   | Name of the :ref:`fallback action           | a
    |                            | <fallback-actions>`                         | a
    |                            | to be called if the confidence of intent    | a
    |                            | or action is below the respective threshold | a
    +----------------------------+---------------------------------------------+ a
 a
    You can also configure the ``FallbackPolicy`` in your python code: a
 a
    .. code-block:: python a
 a
       from rasa.core.policies.fallback import FallbackPolicy a
       from rasa.core.policies.keras_policy import TEDPolicy a
       from rasa.core.agent import Agent a
 a
       fallback = FallbackPolicy(fallback_action_name="action_default_fallback", a
                                 core_threshold=0.3, a
                                 nlu_threshold=0.3, a
                                 ambiguity_threshold=0.1) a
 a
       agent = Agent("domain.yml", policies=[TEDPolicy(), fallback]) a
 a
    .. note:: a
 a
       You can include either the ``FallbackPolicy`` or the a
       ``TwoStageFallbackPolicy`` in your configuration, but not both. a
 a
 a
Two-Stage Fallback Policy a
^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The ``TwoStageFallbackPolicy`` handles low NLU confidence in multiple stages a
by trying to disambiguate the user input. a
 a
- If an NLU prediction has a low confidence score or is not significantly higher a
  than the second highest ranked prediction, the user is asked to affirm a
  the classification of the intent. a
 a
    - If they affirm, the story continues as if the intent was classified a
      with high confidence from the beginning. a
    - If they deny, the user is asked to rephrase their message. a
 a
- Rephrasing a
 a
    - If the classification of the rephrased intent was confident, the story a
      continues as if the user had this intent from the beginning. a
    - If the rephrased intent was not classified with high confidence, the user a
      is asked to affirm the classified intent. a
 a
- Second affirmation a
 a
    - If the user affirms the intent, the story continues as if the user had a
      this intent from the beginning. a
    - If the user denies, the original intent is classified as the specified a
      ``deny_suggestion_intent_name``, and an ultimate fallback action a
      is triggered (e.g. a handoff to a human). a
 a
**Configuration:** a
 a
    To use the ``TwoStageFallbackPolicy``, include the following in your a
    policy configuration. a
 a
    .. code-block:: yaml a
 a
        policies: a
          - name: TwoStageFallbackPolicy a
            nlu_threshold: 0.3 a
            ambiguity_threshold: 0.1 a
            core_threshold: 0.3 a
            fallback_core_action_name: "action_default_fallback" a
            fallback_nlu_action_name: "action_default_fallback" a
            deny_suggestion_intent_name: "out_of_scope" a
 a
    +-------------------------------+------------------------------------------+ a
    | ``nlu_threshold``             | Min confidence needed to accept an NLU   | a
    |                               | prediction                               | a
    +-------------------------------+------------------------------------------+ a
    | ``ambiguity_threshold``       | Min amount by which the confidence of the| a
    |                               | top intent must exceed that of the second| a
    |                               | highest ranked intent.                   | a
    +-------------------------------+------------------------------------------+ a
    | ``core_threshold``            | Min confidence needed to accept an action| a
    |                               | prediction from Rasa Core                | a
    +-------------------------------+------------------------------------------+ a
    | ``fallback_core_action_name`` | Name of the :ref:`fallback action        | a
    |                               | <fallback-actions>`                      | a
    |                               | to be called if the confidence of Rasa   | a
    |                               | Core action prediction is below the      | a
    |                               | ``core_threshold``. This action is       |   a
    |                               | to propose the recognized intents        | a
    +-------------------------------+------------------------------------------+ a
    | ``fallback_nlu_action_name``  | Name of the :ref:`fallback action        | a
    |                               | <fallback-actions>`                      | a
    |                               | to be called if the confidence of Rasa   | a
    |                               | NLU intent classification is below the   | a
    |                               | ``nlu_threshold``. This action is called | a
    |                               | when the user denies the second time     | a
    +-------------------------------+------------------------------------------+ a
    |``deny_suggestion_intent_name``| The name of the intent which is used to  | a
    |                               | detect that the user denies the suggested| a
    |                               | intents                                  | a
    +-------------------------------+------------------------------------------+ a
 a
    .. note:: a
 a
      You can include either the ``FallbackPolicy`` or the a
      ``TwoStageFallbackPolicy`` in your configuration, but not both. a
 a
 a
.. _form-policy: a
 a
Form Policy a
^^^^^^^^^^^ a
 a
The ``FormPolicy`` is an extension of the ``MemoizationPolicy`` which a
handles the filling of forms. Once a ``FormAction`` is called, the a
``FormPolicy`` will continually predict the ``FormAction`` until all required a
slots in the form are filled. For more information, see :ref:`forms`. a
 a