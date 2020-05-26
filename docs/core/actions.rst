:desc: Learn about about how to write your own custom actions with the a
       open source Rasa framework to be able to interact with the external a
       world - ranging from databases to third-party APIs. a
 a
.. _actions: a
 a
Actions a
======= a
 a
.. edit-link:: a
 a
Actions are the things your bot runs in response to user input. a
There are four kinds of actions in Rasa: a
 a
 1. **Utterance actions**: start with ``utter_`` and send a specific message a
    to the user. a
 2. **Retrieval actions**: start with ``respond_`` and send a message selected by a retrieval model. a
 3. **Custom actions**: run arbitrary code and send any number of messages (or none). a
 4. **Default actions**: e.g. ``action_listen``, ``action_restart``, a
    ``action_default_fallback``. a
 a
.. contents:: a
   :local: a
 a
Utterance Actions a
----------------- a
 a
To define an utterance action (``ActionUtterTemplate``), add a response to the domain file a
that starts with ``utter_``: a
 a
.. code-block:: yaml a
 a
    responses: a
      utter_my_message: a
        - "this is what I want my action to say!" a
 a
It is conventional to start the name of an utterance action with ``utter_``. a
If this prefix is missing, you can still use the response in your custom a
actions, but the response can not be directly predicted as its own action. a
See :ref:`responses` for more details. a
 a
If you use an external NLG service, you don't need to specify the a
responses in the domain, but you still need to add the utterance names a
to the actions list of the domain. a
 a
 a
Retrieval Actions a
----------------- a
 a
Retrieval actions make it easier to work with a large number of similar intents like chitchat and FAQs. a
See :ref:`retrieval-actions` to learn more. a
 a
.. _custom-actions: a
 a
Custom Actions a
-------------- a
 a
An action can run any code you want. Custom actions can turn on the lights, a
add an event to a calendar, check a user's bank balance, or anything a
else you can imagine. a
 a
Rasa will call an endpoint you can specify, when a custom action is a
predicted. This endpoint should be a webserver that reacts to this a
call, runs the code and optionally returns information to modify a
the dialogue state. a
 a
To specify, your action server use the ``endpoints.yml``: a
 a
.. code-block:: yaml a
 a
   action_endpoint: a
     url: "http://localhost:5055/webhook" a
 a
And pass it to the scripts using ``--endpoints endpoints.yml``. a
 a
You can create an action server in node.js, .NET, java, or any a
other language and define your actions there - but we provide a
a small python SDK to make development there even easier. a
 a
.. note:: a
 a
    Rasa uses a ticket lock mechanism to ensure incoming messages from the same a
    conversation ID do not interfere with each other and are processed in the right a
    order. If you expect your custom action to take more than 60 seconds to run, please a
    set the ``TICKET_LOCK_LIFETIME`` environment variable to your expected value. a
 a
Custom Actions Written in Python a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
For actions written in python, we have a convenient :ref:`rasa-sdk` which starts a
this action server for you. a
 a
Execute Actions in Other Code a
----------------------------- a
 a
Rasa will send an HTTP ``POST`` request to your server containing a
information on which action to run. Furthermore, this request will contain all a
information about the conversation. :ref:`action-server` shows the detailed API spec. a
 a
As a response to the action call from Rasa, you can modify the tracker, a
e.g. by setting slots and send responses back to the user. a
All of the modifications are done using events. a
There is a list of all possible event types in :ref:`events`. a
 a
.. _default-actions: a
 a
Default Actions a
--------------- a
 a
The available default actions are: a
 a
+-----------------------------------+------------------------------------------------+ a
| ``action_listen``                 | Stop predicting more actions and wait for user | a
|                                   | input.                                         | a
+-----------------------------------+------------------------------------------------+ a
| ``action_restart``                | Reset the whole conversation. Can be triggered | a
|                                   | during a conversation by entering ``/restart`` | a
|                                   | if the :ref:`mapping-policy` is included in    | a
|                                   | the policy configuration.                      | a
+-----------------------------------+------------------------------------------------+ a
| ``action_session_start``          | Start a new conversation session. Take all set | a
|                                   | slots, mark the beginning of a new conversation| a
|                                   | session and re-apply the existing ``SlotSet``  | a
|                                   | events. This action is triggered automatically | a
|                                   | after an inactivity period defined by the      | a
|                                   | ``session_expiration_time`` parameter in the   | a
|                                   | domain's :ref:`session_config`. Can be         | a
|                                   | triggered manually during a conversation by    | a
|                                   | entering ``/session_start``. All conversations | a
|                                   | begin with an ``action_session_start``.        | a
+-----------------------------------+------------------------------------------------+ a
| ``action_default_fallback``       | Undo the last user message (as if the user did | a
|                                   | not send it and the bot did not react) and     | a
|                                   | utter a message that the bot did not           | a
|                                   | understand. See :ref:`fallback-actions`.       | a
+-----------------------------------+------------------------------------------------+ a
| ``action_deactivate_form``        | Deactivate the active form and reset the       | a
|                                   | requested slot.                                | a
|                                   | See also :ref:`section_unhappy`.               | a
+-----------------------------------+------------------------------------------------+ a
| ``action_revert_fallback_events`` | Revert events that occurred during the         | a
|                                   | TwoStageFallbackPolicy.                        | a
|                                   | See :ref:`fallback-actions`.                   | a
+-----------------------------------+------------------------------------------------+ a
| ``action_default_ask_affirmation``| Ask the user to affirm their intent.           | a
|                                   | It is suggested to overwrite this default      | a
|                                   | action with a custom action to have more       | a
|                                   | meaningful prompts.                            | a
+-----------------------------------+------------------------------------------------+ a
| ``action_default_ask_rephrase``   | Ask the user to rephrase their intent.         | a
+-----------------------------------+------------------------------------------------+ a
| ``action_back``                   | Undo the last user message (as if the user did | a
|                                   | not send it and the bot did not react).        | a
|                                   | Can be triggered during a conversation by      | a
|                                   | entering ``/back`` if the MappingPolicy is     | a
|                                   | included in the policy configuration.          | a
+-----------------------------------+------------------------------------------------+ a
 a
All the default actions can be overridden. To do so, add the action name a
to the list of actions in your domain: a
 a
.. code-block:: yaml a
 a
  actions: a
  - action_default_ask_affirmation a
 a
Rasa will then call your action endpoint and treat it as every other a
custom action. a
 a