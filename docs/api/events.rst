:desc: Use events in open source library Rasa Core to support functionalities a
       like resetting slots, scheduling reminder or pausing a conversation. a
 a
.. _events: a
 a
Events a
====== a
 a
.. edit-link:: a
 a
Conversations in Rasa are represented as a sequence of events. a
This page lists the event types defined in Rasa Core. a
 a
.. note:: a
    If you are using the Rasa SDK to write custom actions in python, a
    you need to import the events from ``rasa_sdk.events``, not from a
    ``rasa.core.events``. If you are writing actions in another language, a
    your events should be formatted like the JSON objects on this page. a
 a
 a
 a
.. contents:: a
   :local: a
 a
General Purpose Events a
---------------------- a
 a
Set a Slot a
~~~~~~~~~~ a
 a
:Short: Event to set a slot on a tracker a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER SetSlot a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.SlotSet a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: SlotSet.apply_to a
 a
 a
Restart a conversation a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Resets anything logged on the tracker. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER Restarted a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.Restarted a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: Restarted.apply_to a
 a
 a
Reset all Slots a
~~~~~~~~~~~~~~~ a
 a
:Short: Resets all the slots of a conversation. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER AllSlotsReset a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.AllSlotsReset a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: AllSlotsReset.apply_to a
 a
 a
Schedule a reminder a
~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Schedule an intent to be triggered in the future. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :lines: 1- a
      :start-after: # DOCS MARKER ReminderScheduled a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.ReminderScheduled a
 a
:Effect: a
    When added to a tracker, Rasa Core will schedule the intent (and entities) to be a
    triggered in the future, in place of a user input. You can link a
    this intent to an action of your choice using the :ref:`mapping-policy`. a
 a
 a
Cancel a reminder a
~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Cancel one or more reminders. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :lines: 1- a
      :start-after: # DOCS MARKER ReminderCancelled a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.ReminderCancelled a
 a
:Effect: a
    When added to a tracker, Rasa Core will cancel any outstanding reminders that a
    match the ``ReminderCancelled`` event. For example, a
 a
    - ``ReminderCancelled(intent="greet")`` cancels all reminders with intent ``greet`` a
    - ``ReminderCancelled(entities={...})`` cancels all reminders with the given entities a
    - ``ReminderCancelled("...")`` cancels the one unique reminder with the given name a
    - ``ReminderCancelled()`` cancels all reminders a
 a
 a
Pause a conversation a
~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Stops the bot from responding to messages. Action prediction a
        will be halted until resumed. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER ConversationPaused a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.ConversationPaused a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: ConversationPaused.apply_to a
 a
 a
Resume a conversation a
~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Resumes a previously paused conversation. The bot will start a
        predicting actions again. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER ConversationResumed a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.ConversationResumed a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: ConversationResumed.apply_to a
 a
 a
Force a followup action a
~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Instead of predicting the next action, force the next action a
        to be a fixed one. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER FollowupAction a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.FollowupAction a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: FollowupAction.apply_to a
 a
 a
Automatically tracked events a
---------------------------- a
 a
 a
User sent message a
~~~~~~~~~~~~~~~~~ a
 a
:Short: Message a user sent to the bot. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :lines: 1- a
      :start-after: # DOCS MARKER UserUttered a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.UserUttered a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: UserUttered.apply_to a
 a
 a
Bot responded message a
~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Message a bot sent to the user. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER BotUttered a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.BotUttered a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: BotUttered.apply_to a
 a
 a
Undo a user message a
~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Undoes all side effects that happened after the last user message a
        (including the ``user`` event of the message). a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER UserUtteranceReverted a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.UserUtteranceReverted a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: UserUtteranceReverted.apply_to a
 a
 a
Undo an action a
~~~~~~~~~~~~~~ a
 a
:Short: Undoes all side effects that happened after the last action a
        (including the ``action`` event of the action). a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER ActionReverted a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.ActionReverted a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: ActionReverted.apply_to a
 a
 a
Log an executed action a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Logs an action the bot executed to the conversation. Events that a
        action created are logged separately. a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER ActionExecuted a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.ActionExecuted a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: ActionExecuted.apply_to a
 a
Start a new conversation session a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Marks the beginning of a new conversation session. Resets the tracker and a
        triggers an ``ActionSessionStart`` which by default applies the existing a
        ``SlotSet`` events to the new session. a
 a
:JSON: a
    .. literalinclude:: ../../tests/core/test_events.py a
      :start-after: # DOCS MARKER SessionStarted a
      :dedent: 4 a
      :end-before: # DOCS END a
:Class: a
    .. autoclass:: rasa.core.events.SessionStarted a
 a
:Effect: a
    When added to a tracker, this is the code used to update the tracker: a
 a
    .. literalinclude:: ../../rasa/core/events/__init__.py a
      :dedent: 4 a
      :pyobject: SessionStarted.apply_to a
 a