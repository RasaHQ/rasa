:desc: Learn about about how to write your own custom actions with the
       open source Rasa framework to be able to interact with the external
       world - ranging from databases to third-party APIs.

.. _actions:

Actions
=======

.. edit-link::

Actions are the things your bot runs in response to user input.
There are four kinds of actions in Rasa:

 1. **Utterance actions**: start with ``utter_`` and send a specific message
    to the user
 2. **Retrieval actions**: start with ``respond_`` and send a message selected by a retrieval model
 3. **Custom actions**: run arbitrary code and send any number of messages (or none).
 4. **Default actions**: e.g. ``action_listen``, ``action_restart``,
    ``action_default_fallback``

.. contents::
   :local:

Utterance Actions
-----------------

To define an utterance action (``ActionUtterTemplate``), add an utterance template to the domain file
that starts with ``utter_``:

.. code-block:: yaml

    templates:
      utter_my_message:
        - "this is what I want my action to say!"

It is conventional to start the name of an utterance action with ``utter_``.
If this prefix is missing, you can still use the template in your custom
actions, but the template can not be directly predicted as its own action.
See :ref:`responses` for more details.

If you use an external NLG service, you don't need to specify the
templates in the domain, but you still need to add the utterance names
to the actions list of the domain.


Retrieval Actions
-----------------

Retrieval actions make it easier to work with a large number of similar intents like chitchat and FAQs.
See :ref:`retrieval-actions` to learn moree.

.. _custom-actions:

Custom Actions
--------------

An action can run any code you want. Custom actions can turn on the lights,
add an event to a calendar, check a user's bank balance, or anything
else you can imagine.

Rasa will call an endpoint you can specify, when a custom action is
predicted. This endpoint should be a webserver that reacts to this
call, runs the code and optionally returns information to modify
the dialogue state.

To specify, your action server use the ``endpoints.yml``:

.. code-block:: yaml

   action_endpoint:
     url: "http://localhost:5055/webhook"

And pass it to the scripts using ``--endpoints endpoints.yml``.

You can create an action server in node.js, .NET, java, or any
other language and define your actions there - but we provide
a small python SDK to make development there even easier.

.. note::

    Rasa uses a ticket lock mechanism to ensure incoming messages from the same
    conversation ID do not interfere with each other and are processed in the right
    order. If you expect your custom action to take more than 60 seconds to run, please
    set the ``TICKET_LOCK_LIFETIME`` environment variable to your expected value.

Custom Actions Written in Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For actions written in python, we have a convenient :ref:`rasa-sdk` which starts
this action server for you.

Execute Actions in Other Code
-----------------------------

Rasa will send an HTTP ``POST`` request to your server containing
information on which action to run. Furthermore, this request will contain all
information about the conversation. :ref:`action-server` shows the detailed API spec.

As a response to the action call from Rasa, you can modify the tracker,
e.g. by setting slots and send responses back to the user.
All of the modifications are done using events.
There is a list of all possible event types in :ref:`events`.

Proactively Reaching Out to the User Using Actions
--------------------------------------------------

You may want to proactively reach out to the user,
for example to display the output of a long running background operation
or notify the user of an external event.

To do so, you can ``POST`` to this
`endpoint <../../api/http-api/#operation/executeConversationAction>`_ ,
specifying the action which should be run for a specific user in the request body. Use the
``output_channel`` query parameter to specify which output
channel should be used to communicate the assistant's responses back to the user.
If your message is static, you can define an ``utter_`` action in your domain file with
a corresponding template. If you need more control, add a custom action in your
domain and implement the required steps in your action server. Any messages which are
dispatched in the custom action will be forwarded to the specified output channel.


Proactively reaching out to the user is dependent on the abilities of a channel and
hence not supported by every channel. If your channel does not support it, consider
using the :ref:`callbackInput` channel to send messages to a webhook.


.. note::

   Running an action in a conversation changes the conversation history and affects the
   assistant's next predictions. If you don't want this to happen, make sure that your action
   reverts itself by appending a ``ActionReverted`` event to the end of the
   conversation tracker.

.. _default-actions:

Default Actions
---------------

The available default actions are:

+-----------------------------------+------------------------------------------------+
| ``action_listen``                 | Stop predicting more actions and wait for user |
|                                   | input.                                         |
+-----------------------------------+------------------------------------------------+
| ``action_restart``                | Reset the whole conversation. Can be triggered |
|                                   | during a conversation by entering ``/restart`` |
|                                   | if the :ref:`mapping-policy` is included in    |
|                                   | the policy configuration.                      |
+-----------------------------------+------------------------------------------------+
| ``action_session_start``          | Start a new conversation session. Take all set |
|                                   | slots, mark the beginning of a new conversation|
|                                   | session and re-apply the existing ``SlotSet``  |
|                                   | events. This action is triggered automatically |
|                                   | after an inactivity period defined by the      |
|                                   | ``session_expiration_time`` parameter in the   |
|                                   | domain's :ref:`session_config`. Can be         |
|                                   | triggered manually during a conversation by    |
|                                   | entering ``/session_start``. All conversations |
|                                   | begin with an ``action_session_start``.        |
+-----------------------------------+------------------------------------------------+
| ``action_default_fallback``       | Undo the last user message (as if the user did |
|                                   | not send it and the bot did not react) and     |
|                                   | utter a message that the bot did not           |
|                                   | understand. See :ref:`fallback-actions`.       |
+-----------------------------------+------------------------------------------------+
| ``action_deactivate_form``        | Deactivate the active form and reset the       |
|                                   | requested slot.                                |
|                                   | See also :ref:`section_unhappy`.               |
+-----------------------------------+------------------------------------------------+
| ``action_revert_fallback_events`` | Revert events that occurred during the         |
|                                   | TwoStageFallbackPolicy.                        |
|                                   | See :ref:`fallback-actions`.                   |
+-----------------------------------+------------------------------------------------+
| ``action_default_ask_affirmation``| Ask the user to affirm their intent.           |
|                                   | It is suggested to overwrite this default      |
|                                   | action with a custom action to have more       |
|                                   | meaningful prompts.                            |
+-----------------------------------+------------------------------------------------+
| ``action_default_ask_rephrase``   | Ask the user to rephrase their intent.         |
+-----------------------------------+------------------------------------------------+
| ``action_back``                   | Undo the last user message (as if the user did |
|                                   | not send it and the bot did not react).        |
|                                   | Can be triggered during a conversation by      |
|                                   | entering ``/back`` if the MappingPolicy is     |
|                                   | included in the policy configuration.          |
+-----------------------------------+------------------------------------------------+

All the default actions can be overridden. To do so, add the action name
to the list of actions in your domain:

.. code-block:: yaml

  actions:
  - action_default_ask_affirmation

Rasa will then call your action endpoint and treat it as every other
custom action.
