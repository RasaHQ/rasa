.. _featurization:

Featurization
==============

In order to apply machine learning algorithms to conversational AI, we need to build up vector representations of conversations.


We use the ``X, y`` notation that's common for supervised learning, where ``X`` is a matrix of shape ``(num_data_points, data_dimension)``, and ``y`` is a 1D array of length ``num_data_points`` containing the target class labels.

The target labels correspond to actions taken by the bot.
If the domain defines the possible ``actions`` ``[ActionGreet, ActionListen]`` then a label 0 indicates a greeting and 1 indicates a listen.

The rows in ``X`` correspond to the state of the conversation just before the action was taken.

Featurising a single state works like this:
the tracker provides a bag of ``active_features`` comprising:

- what the last action was (e.g. ``prev_action_listen``)
- features indicating intents and entities, if this is the first state in a turn, e.g. it's the first action we will take after parsing the user's message. (e.g. ``[intent_restaurant_search, entity_cuisine]`` )
- features indicating which slots are currently defined, e.g. ``slot_location`` if the user previously mentioned the area they're searching for restaurants.
- features indicating the results of any API calls stored in slots, e.g. ``slot_matches``

All of these features are represented in a binary vector which just indicates if they're present.
 e.g. ``[0 0 1 1 0 1 ...]``

To recover the bag of features from a vector ``vec``, you can call ``domain.reverse_binary_encoded_features(vec)``.
This is very useful for debugging.

History
-------

It's often useful to include a bit more history than just the current state in memory.
The parameter ``max_history`` defines how many states go into defining each row in ``X``. 

Hence the statement above that ``X`` is 2D is actually false, it has shape ``(num_states, max_history, num_features)``.
For most algorithms you want a flat feature vector, so you will have to reshape this to ``(num_states, max_history * num_features)``.