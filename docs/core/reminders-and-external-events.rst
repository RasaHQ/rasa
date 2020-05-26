:desc: Learn how to use external events and schedule reminders. a
 a
.. _reminders-and-external-events: a
 a
Reminders and External Events a
============================= a
 a
.. edit-link:: a
 a
The ``ReminderScheduled`` event and the a
`trigger_intent endpoint <../../api/http-api/#operation/triggerConversationIntent>`_ let your assistant remind you a
about things after a given period of time, or to respond to external events (other applications, sensors, etc.). a
You can find a full example assistant that implements these features a
`here <https://github.com/RasaHQ/rasa/blob/master/examples/reminderbot/README.md>`_. a
 a
.. contents:: a
   :local: a
 a
.. _reminders: a
 a
Reminders a
--------- a
 a
Instead of an external sensor, you might just want to be reminded about something after a certain amount of time. a
For this, Rasa provides the special event ``ReminderScheduled``, and another event, ``ReminderCancelled``, to unschedule a reminder. a
 a
.. _scheduling-reminders-guide: a
 a
Scheduling Reminders a
^^^^^^^^^^^^^^^^^^^^ a
 a
Let's say you want your assistant to remind you to call a friend in 5 seconds. a
(You probably want some longer time span, but for the sake of testing, let it be 5 seconds.) a
Thus, we define an intent ``ask_remind_call`` with some NLU data, a
 a
.. code-block:: md a
 a
  ## intent:ask_remind_call a
  - remind me to call [Albert](name) a
  - remind me to call [Susan](name) a
  - later I have to call [Daksh](name) a
  - later I have to call [Anna](name) a
  ... a
 a
and connect this intent with a new custom action ``action_set_reminder``. a
We could make this connection by providing training stories (recommended for more complex assistants), or using the :ref:`mapping-policy`. a
 a
The custom action ``action_set_reminder`` should schedule a reminder that, 5 seconds later, triggers an intent ``EXTERNAL_reminder`` with all the entities that the user provided in his/her last message (similar to an external event): a
 a
.. literalinclude:: ../../examples/reminderbot/actions.py a
   :pyobject: ActionSetReminder a
 a
Note that this requires the ``datetime`` and ``rasa_sdk.events`` packages. a
 a
Finally, we define another custom action ``action_react_to_reminder`` and link it to the ``EXTERNAL_reminder`` intent: a
 a
.. code-block:: md a
 a
  - EXTERNAL_reminder: a
    triggers: action_react_to_reminder a
 a
where the ``action_react_to_reminder`` is a
 a
.. literalinclude:: ../../examples/reminderbot/actions.py a
   :pyobject: ActionReactToReminder a
 a
Instead of a custom action, we could also have used a simple response template. a
But here we want to make use of the fact that the reminder can carry entities, and we can process the entities in this custom action. a
 a
.. warning:: a
 a
  Reminders are cancelled whenever you shutdown your Rasa server. a
 a
.. warning:: a
 a
  Reminders currently (Rasa 1.8) don't work in `rasa shell`. a
  You have to test them with a a
  `running Rasa X server <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-script/>`_ instead. a
 a
.. note:: a
 a
   Proactively reaching out to the user is dependent on the abilities of a channel and a
   hence not supported by every channel. If your channel does not support it, consider a
   using the :ref:`callbackInput` channel to send messages to a `webhook <https://en.wikipedia.org/wiki/Webhook>`_. a
 a
.. _cancelling-reminders-guide: a
 a
Cancelling Reminders a
^^^^^^^^^^^^^^^^^^^^ a
 a
Sometimes the user may want to cancel a reminder that he has scheduled earlier. a
A simple way of adding this functionality to your assistant is to create an intent ``ask_forget_reminders`` and let your assistant respond to it with a custom action such as a
 a
.. literalinclude:: ../../examples/reminderbot/actions.py a
   :pyobject: ForgetReminders a
 a
Here, ``ReminderCancelled()`` simply cancels all the reminders that are currently scheduled. a
Alternatively, you may provide some parameters to narrow down the types of reminders that you want to cancel. a
For example, a
 a
    - ``ReminderCancelled(intent="greet")`` cancels all reminders with intent ``greet`` a
    - ``ReminderCancelled(entities={...})`` cancels all reminders with the given entities a
    - ``ReminderCancelled("...")`` cancels the one unique reminder with the given name "``...``" that you supplied a
      during its creation a
 a
.. _external-event-guide: a
 a
External Events a
--------------- a
 a
Let's say you want to send a message from some other device to change the course of an ongoing conversation. a
For example, some moisture-sensor attached to a Raspberry Pi should inform your personal assistant that your favourite a
plant needs watering, and your assistant should then relay this message to you. a
 a
To do this, your Raspberry Pi needs to send a message to the `trigger_intent endpoint <../../api/http-api/#operation/triggerConversationIntent>`_ of your conversation. a
As the name says, this injects a user intent (possibly with entities) into your conversation. a
So for Rasa it is almost as if you had entered a message that got classified with this intent and these entities. a
Rasa then needs to respond to this input with an action such as ``action_warn_dry``. a
The easiest and most reliable way to connect this action with the intent is via the :ref:`mapping-policy`. a
 a
.. _getting-conversation-id: a
 a
Getting the Conversation ID a
^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The first thing we need is the Session ID of the conversation that your sensor should send a notification to. a
An easy way to get this is to define a custom action (see :ref:`custom-actions`) that displays the ID in the conversation. a
For example: a
 a
.. literalinclude:: ../../examples/reminderbot/actions.py a
   :pyobject: ActionTellID a
 a
In addition, we also declare an intent ``ask_id``, define some NLU data for it, and add both ``action_tell_id`` and a
``ask_id`` to the domain file, where we specify that one should trigger the other: a
 a
.. code-block:: md a
 a
  intents: a
    - ask_id: a
      triggers: action_tell_id a
 a
Now, when you ask "What is the ID of this conversation?", the assistant replies with something like "The ID of this a
conversation is: 38cc25d7e23e4dde800353751b7c2d3e". a
 a
If you want your assistant to link to the Raspberry Pi automatically, you will have to write a custom action that a
informs the Pi about the conversation id when your conversation starts (see :ref:`custom_session_start`). a
 a
.. _responding_to_external_events: a
 a
Responding to External Events a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
Now that we have our Session ID, we need to prepare the assistant so it responds to messages from the sensor. a
To this end, we define a new intent ``EXTERNAL_dry_plant`` without any NLU data. a
This intent will later be triggered by the external sensor. a
Here, we start the intent name with ``EXTERNAL_`` to indicate that this is not something the user would say, but you can name the intent however you like. a
 a
In the domain file, we now connect the intent ``EXTERNAL_dry_plant`` with another custom action ``action_warn_dry``, e.g. a
 a
.. literalinclude:: ../../examples/reminderbot/actions.py a
   :pyobject: ActionWarnDry a
 a
Now, when you are in a conversation with id ``38cc25d7e23e4dde800353751b7c2d3e``, then running a
 a
.. code-block:: shell a
 a
  curl -H "Content-Type: application/json" -X POST -d '{"name": "EXTERNAL_dry_plant", "entities": {"plant": "Orchid"}}' http://localhost:5005/conversations/38cc25d7e23e4dde800353751b7c2d3e/trigger_intent a
 a
in the terminal will cause your assistant to say "Your Orchid needs some water!". a
 a