:desc: Learn how to use external events and schedule reminders.

.. _reminders-and-external-events:

Reminders and External Events
=============================

.. edit-link::

The ``ReminderScheduled`` event and the
`trigger_intent endpoint <../../api/http-api/#operation/triggerConversationIntent>`_ let your assistant remind you
about things after a given period of time, or to respond to external events (other applications, sensors, etc.).
You can find a full example assistant that implements these features
`here <https://github.com/RasaHQ/rasa/blob/master/examples/reminderbot/README.md>`_.

.. contents::
   :local:

.. _reminders:

Reminders
---------

Instead of an external sensor, you might just want to be reminded about something after a certain amount of time.
For this, Rasa provides the special event ``ReminderScheduled``, and another event, ``ReminderCancelled``, to unschedule a reminder.

.. _scheduling-reminders-guide:

Scheduling Reminders
^^^^^^^^^^^^^^^^^^^^

Let's say you want your assistant to remind you to call a friend in 5 seconds.
(You probably want some longer time span, but for the sake of testing, let it be 5 seconds.)
Thus, we define an intent ``ask_remind_call`` with some NLU data,

.. code-block:: md

  ## intent:ask_remind_call
  - remind me to call [Albert](name)
  - remind me to call [Susan](name)
  - later I have to call [Daksh](name)
  - later I have to call [Anna](name)
  ...

and connect this intent with a new custom action ``action_set_reminder``.
We could make this connection by providing training stories (recommended for more complex assistants), or using the :ref:`mapping-policy`.

The custom action ``action_set_reminder`` should schedule a reminder that, 5 seconds later, triggers an intent ``EXTERNAL_reminder`` with all the entities that the user provided in his/her last message (similar to an external event):

.. literalinclude:: ../../examples/reminderbot/actions.py
   :pyobject: ActionSetReminder

Note that this requires the ``datetime`` and ``rasa_sdk.events`` packages.

Finally, we define another custom action ``action_react_to_reminder`` and link it to the ``EXTERNAL_reminder`` intent:

.. code-block:: md

  - EXTERNAL_reminder:
    triggers: action_react_to_reminder

where the ``action_react_to_reminder`` is

.. literalinclude:: ../../examples/reminderbot/actions.py
   :pyobject: ActionReactToReminder

Instead of a custom action, we could also have used a simple response template.
But here we want to make use of the fact that the reminder can carry entities, and we can process the entities in this custom action.

.. warning::

  Reminders are cancelled whenever you shutdown your Rasa server.

.. warning::

  Reminders currently (Rasa 1.8) don't work in `rasa shell`.
  You have to test them with a
  `running Rasa X server <https://rasa.com/docs/rasa-x/installation-and-setup/docker-compose-script/>`_ instead.

.. note::

   Proactively reaching out to the user is dependent on the abilities of a channel and
   hence not supported by every channel. If your channel does not support it, consider
   using the :ref:`callbackInput` channel to send messages to a `webhook <https://en.wikipedia.org/wiki/Webhook>`_.

.. _cancelling-reminders-guide:

Cancelling Reminders
^^^^^^^^^^^^^^^^^^^^

Sometimes the user may want to cancel a reminder that he has scheduled earlier.
A simple way of adding this functionality to your assistant is to create an intent ``ask_forget_reminders`` and let your assistant respond to it with a custom action such as

.. literalinclude:: ../../examples/reminderbot/actions.py
   :pyobject: ForgetReminders

Here, ``ReminderCancelled()`` simply cancels all the reminders that are currently scheduled.
Alternatively, you may provide some parameters to narrow down the types of reminders that you want to cancel.
For example,

    - ``ReminderCancelled(intent="greet")`` cancels all reminders with intent ``greet``
    - ``ReminderCancelled(entities={...})`` cancels all reminders with the given entities
    - ``ReminderCancelled("...")`` cancels the one unique reminder with the given name "``...``" that you supplied
      during its creation

.. _external-event-guide:

External Events
---------------

Let's say you want to send a message from some other device to change the course of an ongoing conversation.
For example, some moisture-sensor attached to a Raspberry Pi should inform your personal assistant that your favorite
plant needs watering, and your assistant should then relay this message to you.

To do this, your Raspberry Pi needs to send a message to the `trigger_intent endpoint <../../api/http-api/#operation/triggerConversationIntent>`_ of your conversation.
As the name says, this injects a user intent (possibly with entities) into your conversation.
So for Rasa it is almost as if you had entered a message that got classified with this intent and these entities.
Rasa then needs to respond to this input with an action such as ``action_warn_dry``.
The easiest and most reliable way to connect this action with the intent is via the :ref:`mapping-policy`.

.. _getting-conversation-id:

Getting the Conversation ID
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first thing we need is the Session ID of the conversation that your sensor should send a notification to.
An easy way to get this is to define a custom action (see :ref:`custom-actions`) that displays the ID in the conversation.
For example:

.. literalinclude:: ../../examples/reminderbot/actions.py
   :pyobject: ActionTellID

In addition, we also declare an intent ``ask_id``, define some NLU data for it, and add both ``action_tell_id`` and
``ask_id`` to the domain file, where we specify that one should trigger the other:

.. code-block:: md

  intents:
    - ask_id:
      triggers: action_tell_id

Now, when you ask "What is the ID of this conversation?", the assistant replies with something like "The ID of this
conversation is: 38cc25d7e23e4dde800353751b7c2d3e".

If you want your assistant to link to the Raspberry Pi automatically, you will have to write a custom action that
informs the Pi about the conversation id when your conversation starts (see :ref:`custom_session_start`).

.. _responding_to_external_events:

Responding to External Events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have our Session ID, we need to prepare the assistant so it responds to messages from the sensor.
To this end, we define a new intent ``EXTERNAL_dry_plant`` without any NLU data.
This intent will later be triggered by the external sensor.
Here, we start the intent name with ``EXTERNAL_`` to indicate that this is not something the user would say, but you can name the intent however you like.

In the domain file, we now connect the intent ``EXTERNAL_dry_plant`` with another custom action ``action_warn_dry``, e.g.

.. literalinclude:: ../../examples/reminderbot/actions.py
   :pyobject: ActionWarnDry

Now, when you are in a conversation with id ``38cc25d7e23e4dde800353751b7c2d3e``, then running

.. code-block:: shell

  curl -H "Content-Type: application/json" -X POST -d '{"name": "EXTERNAL_dry_plant", "entities": {"plant": "Orchid"}}' http://localhost:5005/conversations/38cc25d7e23e4dde800353751b7c2d3e/trigger_intent

in the terminal will cause your assistant to say "Your Orchid needs some water!".
