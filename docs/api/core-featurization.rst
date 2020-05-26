:desc: Find out how to apply machine learning algorithms to conversational AI a
       using vector representations of conversations with Rasa. a
 a
.. _featurization_conversations: a
 a
Featurization of Conversations a
============================== a
 a
.. edit-link:: a
 a
In order to apply machine learning algorithms to conversational AI, we need a
to build up vector representations of conversations. a
 a
Each story corresponds to a tracker which consists of the states of the a
conversation just before each action was taken. a
 a
 a
State Featurizers a
^^^^^^^^^^^^^^^^^ a
Every event in a trackers history creates a new state (e.g. running a bot a
action, receiving a user message, setting slots). Featurizing a single state a
of the tracker has a couple steps: a
 a
1. **Tracker provides a bag of active features**: a
    - features indicating intents and entities, if this is the first a
      state in a turn, e.g. it's the first action we will take after a
      parsing the user's message. (e.g. a
      ``[intent_restaurant_search, entity_cuisine]`` ) a
    - features indicating which slots are currently defined, e.g. a
      ``slot_location`` if the user previously mentioned the area a
      they're searching for restaurants. a
    - features indicating the results of any API calls stored in a
      slots, e.g. ``slot_matches`` a
    - features indicating what the last action was (e.g. a
      ``prev_action_listen``) a
 a
2. **Convert all the features into numeric vectors**: a
 a
        We use the ``X, y`` notation that's common for supervised learning, a
        where ``X`` is an array of shape a
        ``(num_data_points, time_dimension, num_input_features)``, a
        and ``y`` is an array of shape ``(num_data_points, num_bot_features)`` a
        or ``(num_data_points, time_dimension, num_bot_features)`` a
        containing the target class labels encoded as one-hot vectors. a
 a
        The target labels correspond to actions taken by the bot. a
        To convert the features into vector format, there are different a
        featurizers available: a
 a
        - ``BinarySingleStateFeaturizer`` creates a binary one-hot encoding: a
            The vectors ``X, y`` indicate a presence of a certain intent, a
            entity, previous action or slot e.g. ``[0 0 1 0 0 1 ...]``. a
 a
        - ``LabelTokenizerSingleStateFeaturizer`` creates a vector a
            based on the feature label: a
            All active feature labels (e.g. ``prev_action_listen``) are split a
            into tokens and represented as a bag-of-words. For example, actions a
            ``utter_explain_details_hotel`` and a
            ``utter_explain_details_restaurant`` will have 3 features in a
            common, and differ by a single feature indicating a domain. a
 a
            Labels for user inputs (intents, entities) and bot actions a
            are featurized separately. Each label in the two categories a
            is tokenized on a special character ``split_symbol`` a
            (e.g. ``action_search_restaurant = {action, search, restaurant}``), a
            creating two vocabularies. A bag-of-words representation a
            is then created for each label using the appropriate vocabulary. a
            The slots are featurized as binary vectors, indicating a
            their presence or absence at each step of the dialogue. a
 a
 a
.. note:: a
 a
    If the domain defines the possible ``actions``, a
    ``[ActionGreet, ActionGoodbye]``, a
    ``4`` additional default actions are added: a
    ``[ActionListen(), ActionRestart(), a
    ActionDefaultFallback(), ActionDeactivateForm()]``. a
    Therefore, label ``0`` indicates default action listen, label ``1`` a
    default restart, label ``2`` a greeting and ``3`` indicates goodbye. a
 a
 a
Tracker Featurizers a
^^^^^^^^^^^^^^^^^^^ a
 a
It's often useful to include a bit more history than just the current state a
when predicting an action. The ``TrackerFeaturizer`` iterates over tracker a
states and calls a ``SingleStateFeaturizer`` for each state. There are two a
different tracker featurizers: a
 a
1. Full Dialogue a
---------------- a
 a
``FullDialogueTrackerFeaturizer`` creates numerical representation of a
stories to feed to a recurrent neural network where the whole dialogue a
is fed to a network and the gradient is backpropagated from all time steps. a
Therefore, ``X`` is an array of shape a
``(num_stories, max_dialogue_length, num_input_features)`` and a
``y`` is an array of shape a
``(num_stories, max_dialogue_length, num_bot_features)``. a
The smaller dialogues are padded with ``-1`` for all features, indicating a
no values for a policy. a
 a
2. Max History a
-------------- a
 a
``MaxHistoryTrackerFeaturizer`` creates an array of previous tracker a
states for each bot action or utterance, with the parameter a
``max_history`` defining how many states go into each row in ``X``. a
Deduplication is performed to filter out duplicated turns (bot actions a
or bot utterances) in terms of their previous states. Hence ``X`` a
has shape ``(num_unique_turns, max_history, num_input_features)`` a
and ``y`` is an array of shape ``(num_unique_turns, num_bot_features)``. a
 a
For some algorithms a flat feature vector is needed, so ``X`` a
should be reshaped to a
``(num_unique_turns, max_history * num_input_features)``. If numeric a
target class labels are needed instead of one-hot vectors, use a
``y.argmax(axis=-1)``. a
 a