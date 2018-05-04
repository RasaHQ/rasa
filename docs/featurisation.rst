.. _featurization:

Featurization
==============

In order to apply machine learning algorithms to conversational AI, we need to build up vector representations of conversations.

Each story corresponds to a tracker which consists of the states of the conversation just before each action was taken.

Featurising a single state works like this:
the tracker provides a bag of ``active_features`` comprising:

- what the last action was (e.g. ``prev_action_listen``)
- features indicating intents and entities, if this is the first state in a turn, e.g. it's the first action we will take after parsing the user's message. (e.g. ``[intent_restaurant_search, entity_cuisine]`` )
- features indicating which slots are currently defined, e.g. ``slot_location`` if the user previously mentioned the area they're searching for restaurants.
- features indicating the results of any API calls stored in slots, e.g. ``slot_matches``


Single State Featurizers
^^^^^^^^^^^^^^^^^^^^^^^^

``SingleStateFeaturizer`` converts all of these states into numeric vectors ``X, y``.

We use the ``X, y`` notation that's common for supervised learning,
where ``X`` is a matrix of shape ``(num_data_points, time_dimension, num_features)``,
and ``y`` is an array of length ``num_data_points`` containing the target class labels encoded as one-hot vectors.

The target labels correspond to actions taken by the bot.
.. note::
    If the domain defines the possible ``actions``: ``[ActionGreet, ActionGoodbye]``,
    two additional default actions are added: ``[ActionListen, ActionRestart]``.
    Therefore, label ``0`` indicates default action listen, label ``1`` default restart,
    label ``2`` a greeting and ``3`` indicates goodbye.

Binary
------
``BinarySingleStateFeaturizer`` converts all of these features to a binary vectors ``X, y`` which just indicates if they're present.
 e.g. ``[0 0 1 1 0 1 ...]``

Label Tonizer
-------------


Tracker Featurizers
^^^^^^^^^^^^^^^^^^^


Full Dialogue
-------------



Max History
-----------

It's often useful to include a bit more history than just the current state in memory.
The parameter ``max_history`` defines how many states go into defining each row in ``X``. 

Hence ``X`` has shape ``(num_states, max_history, num_features)``.
For some algorithms you want a flat feature vector, so you will have to reshape this to ``(num_states, max_history * num_features)``.