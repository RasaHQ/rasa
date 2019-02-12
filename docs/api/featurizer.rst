:desc: Find out how to apply machine learning algorithms to conversational AI
       using vector representations of conversations with Rasa Stack.

.. _featurization:

Featurization
==============

In order to apply machine learning algorithms to conversational AI, we need
to build up vector representations of conversations.

Each story corresponds to a tracker which consists of the states of the
conversation just before each action was taken.


State Featurizers
^^^^^^^^^^^^^^^^^
Every event in a trackers history creates a new state (e.g. running a bot
action, receiving a user message, setting slots). Featurizing a single state
of the tracker has a couple steps:

1. **Tracker provides a bag of active features**:
    - features indicating intents and entities, if this is the first
      state in a turn, e.g. it's the first action we will take after
      parsing the user's message. (e.g.
      ``[intent_restaurant_search, entity_cuisine]`` )
    - features indicating which slots are currently defined, e.g.
      ``slot_location`` if the user previously mentioned the area
      they're searching for restaurants.
    - features indicating the results of any API calls stored in
      slots, e.g. ``slot_matches``
    - features indicating what the last action was (e.g.
      ``prev_action_listen``)

2. **Convert all the features into numeric vectors**:

        We use the ``X, y`` notation that's common for supervised learning,
        where ``X`` is an array of shape
        ``(num_data_points, time_dimension, num_input_features)``,
        and ``y`` is an array of shape ``(num_data_points, num_bot_features)``
        or ``(num_data_points, time_dimension, num_bot_features)``
        containing the target class labels encoded as one-hot vectors.

        The target labels correspond to actions taken by the bot.
        To convert the features into vector format, there are different
        feaurizers available:

        - ``BinarySingleStateFeaturizer`` creates a binary one-hot encoding:
            The vectors ``X, y`` indicate a presence of a certain intent,
            entity, previous action or slot e.g. ``[0 0 1 0 0 1 ...]``.

        - ``LabelTokenizerSingleStateFeaturizer`` creates an vector
            based on the feature label:
            All active feature labels (e.g. ``prev_action_listen``) are split
            into tokens and represented as a bag-of-words. For example, actions
            ``utter_explain_details_hotel`` and
            ``utter_explain_details_restaurant`` will have 3 features in
            common, and differ by a single feature indicating a domain.

            Labels for user inputs (intents, entities) and bot actions
            are featurized separately. Each label in the two categories
            is tokenized on a special character ``split_symbol``
            (e.g. ``action_search_restaurant = {action, search, restaurant}``),
            creating two vocabularies. A bag-of-words representation
            is then created for each label using the appropriate vocabulary.
            The slots are featurized as binary vectors, indicating
            their presence or absence at each step of the dialogue.


.. note::

    If the domain defines the possible ``actions``,
    ``[ActionGreet, ActionGoodbye]``,
    ``4`` additional default actions are added:
    ``[ActionListen(), ActionRestart(),
    ActionDefaultFallback(), ActionDeactivateForm()]``.
    Therefore, label ``0`` indicates default action listen, label ``1``
    default restart, label ``2`` a greeting and ``3`` indicates goodbye.


Tracker Featurizers
^^^^^^^^^^^^^^^^^^^

It's often useful to include a bit more history than just the current state
when predicting an action. The ``TrackerFeaturizer`` iterates over tracker
states and calls a ``SingleStateFeaturizer`` for each state. There are two
different tracker featurizers:

1. Full Dialogue
----------------

``FullDialogueTrackerFeaturizer`` creates numerical representation of
stories to feed to a recurrent neural network where the whole dialogue
is fed to a network and the gradient is backpropagated from all time steps.
Therefore, ``X`` is an array of shape
``(num_stories, max_dialogue_length, num_input_features)`` and
``y`` is an array of shape
``(num_stories, max_dialogue_length, num_bot_features)``.
The smaller dialogues are padded with ``-1`` for all features, indicating
no values for a policy.

2. Max History
--------------

``MaxHistoryTrackerFeaturizer`` creates an array of previous tracker
states for each bot action or utterance, with the parameter
``max_history`` defining how many states go into each row in ``X``.
Deduplication is performed to filter out duplicated turns (bot actions
or bot utterances) in terms of their previous states. Hence ``X``
has shape ``(num_unique_turns, max_history, num_input_features)``
and ``y`` is an array of shape ``(num_unique_turns, num_bot_features)``.

For some algorithms a flat feature vector is needed, so ``X``
should be reshaped to
``(num_unique_turns, max_history * num_input_features)``. If numeric
target class labels are needed instead of one-hot vectors, use
``y.argmax(axis=-1)``.


.. include:: ../feedback.inc
