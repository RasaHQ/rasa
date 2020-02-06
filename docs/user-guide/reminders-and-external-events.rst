:desc: Learn how to use external events and schedule reminders.

.. _reminders-and-external-events:

Reminders and external events
=============================

.. edit-link::

In this guide, you will learn how to let your assistant respond to external events (other applications, sensors, etc.) or to remind you about things after a given period of time.

.. contents::
   :local:


.. _external-events:

External events
---------------

Let's say you want to send a message from some other device to change the course of an ongoing conversation.
For example, some moisture-sensor attached to a Raspberry Pi should inform your personal assistant that your favourite
plant needs watering, and your assistant should then relay this message to you.

To do this, your Raspberry Pi needs to send a message to the `trigger_intent endpoint <../../api/http-api/#operation/triggerConversationIntent>`_ of your conversation.
As the name says, this injects a user intent (possibly with entities) into your conversation.
So for Rasa it is almost as if you had entered a message that got classified with this intent and these entities.
Rasa then needs to respond to this input with an action such as ``action_warn_dry``.
The easiest and most reliable way to connect this action with the intent is via the :ref:`mapping-policy`.


.. _getting-conversation-id:

Getting the conversation ID
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first thing we need is the Session ID of the conversation that your sensor should send a notification to.
An easy way to get this is to define a custom action (see :ref:`custom-actions`) that displays the ID in the conversation.
For example:

.. code-block:: python

  class ActionTellID(Action):
    """Informs the user about the conversation ID."""

    def name(self) -> Text:
        return "action_tell_id"

    def run(
        self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        conversation_id = tracker.sender_id

        dispatcher.utter_message(
            f"The ID of this conversation is: '{conversation_id}'."
        )

        return []


In addition, we also declare an intent ``ask_id``, define some NLU data for it, and add both ``action_tell_id`` and ``ask_id`` to the domain file, where we specify that one should trigger the other:

.. code-block:: md

  intents:
    - ask_id:
      triggers: action_tell_id


Now, when you ask "What is the ID of this conversation?", the assistant replies with something like "The ID of this conversation is: 38cc25d7e23e4dde800353751b7c2d3e".
See the ``reminderbot`` example project under ``rasa/examples/reminderbot`` for details.


.. _responding_to_external_events:

Responding to external events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now that we have our Session ID, we need to prepare the assistant so it responds to messages from the sensor.
To this end, we define a new intent ``EXTERNAL_dry_plant`` without any NLU data.
This intent will later be triggered by the external sensor.
Here, we start the intent name with ``EXTERNAL_`` to indicate that this is not something the user would say, but you can name the intent however you like.

In the domain file, we now connect the intent ``EXTERNAL_dry_plant`` with another custom action ``action_warn_dry``, e.g.

.. code-block:: python

  class ActionWarnDry(Action):
    """Informs the user that a plant needs water."""
    def name(self) -> Text:
        return "action_warn_dry"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        plant = next(tracker.get_latest_entity_values("plant"), None) or "plant"
        dispatcher.utter_message(f"Your {plant} needs some water!")

        return []


Now, when you are in a conversation with id ``38cc25d7e23e4dde800353751b7c2d3e``, then running

.. code-block:: shell

  curl -H "Content-Type: application/json" -X POST -d '{"name": "EXTERNAL_dry_plant", "entities": {"plant": "Orchid"}}' http://localhost:5005/conversations/38cc25d7e23e4dde800353751b7c2d3e/trigger_intent


in the terminal will cause your assistant to say "Your Orchid needs some water!".


.. _reminders:

Reminders
---------

Instead of an external sensor, you might just want to be reminded about something after a certain amount of time.
For this, Rasa provides the special event ``ReminderScheduled``.

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

.. code-block:: python

  class ActionSetReminder(Action):
    """Schedules a reminder, supplied with the last message's entities."""

    def name(self) -> Text:
        return "action_set_reminder"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message("I will remind you in 5 seconds.")

        date = datetime.datetime.now() + datetime.timedelta(seconds=5)
        entities = tracker.latest_message.get("entities")

        reminder = ReminderScheduled(
            "EXTERNAL_reminder",
            trigger_date_time=date,
            entities=entities,
            name="my_reminder",
            kill_on_user_message=False,
        )

        return [reminder]


Note, that this requires the ``datetime`` and ``rasa-sdk.events`` packages.
For details, have a look at the ``reminderbot`` example under ``rasa/examples/reminderbot``.

Finally, we define another custom action ``action_react_to_reminder`` and link it to the ``EXTERNAL_reminder`` intent:

.. code-block:: md

  - EXTERNAL_reminder:
    triggers: action_react_to_reminder

where the ``action_react_to_reminder`` is

.. code-block:: python

  class ActionReactToReminder(Action):
    """Reminds the user to call someone."""

    def name(self) -> Text:
        return "action_react_to_reminder"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        name = next(tracker.get_latest_entity_values("name"), None) or "someone"
        dispatcher.utter_message(f"Remember to call {name}!")

        return []

Instead of a custom action, we could also have used a simple response template.
But here we want to make use of the fact that the reminder can carry entities, and we can process the entities in this custom action.

.. warning::

  Reminders are cancelled whenever you shutdown rasa.


.. warning::

  Reminders currently (Rasa 1.7) don't work in `rasa shell`.
  Use `rasa x` instead.

Check out the ``reminderbot`` example project under ``rasa/examples/reminderbot``, and feel free to customize things for your own assistant!
