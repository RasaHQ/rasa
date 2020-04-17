:desc: Use events in open source library Rasa Core to support functionalities a 
       like resetting slots, scheduling reminder or pausing a conversation.

.. _events:

Events a 
======

.. edit-link::

Conversations in Rasa are represented as a sequence of events.
This page lists the event types defined in Rasa Core.

.. note::
    If you are using the Rasa SDK to write custom actions in python,
    you need to import the events from ``rasa_sdk.events``, not from a 
    ``rasa.core.events``. If you are writing actions in another language,
    your events should be formatted like the JSON objects on this page.



.. contents::
   :local:

General Purpose Events a 
----------------------

Set a Slot a 
~~~~~~~~~~

:Short: Event to set a slot on a tracker a 
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER SetSlot a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.SlotSet a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: SlotSet.apply_to a 


Restart a conversation a 
~~~~~~~~~~~~~~~~~~~~~~

:Short: Resets anything logged on the tracker.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER Restarted a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.Restarted a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: Restarted.apply_to a 


Reset all Slots a 
~~~~~~~~~~~~~~~

:Short: Resets all the slots of a conversation.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER AllSlotsReset a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.AllSlotsReset a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: AllSlotsReset.apply_to a 


Schedule a reminder a 
~~~~~~~~~~~~~~~~~~~

:Short: Schedule an intent to be triggered in the future.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :lines: 1-
      :start-after: # DOCS MARKER ReminderScheduled a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.ReminderScheduled a 

:Effect:
    When added to a tracker, Rasa Core will schedule the intent (and entities) to be a 
    triggered in the future, in place of a user input. You can link a 
    this intent to an action of your choice using the :ref:`mapping-policy`.


Cancel a reminder a 
~~~~~~~~~~~~~~~~~~~

:Short: Cancel one or more reminders.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :lines: 1-
      :start-after: # DOCS MARKER ReminderCancelled a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.ReminderCancelled a 

:Effect:
    When added to a tracker, Rasa Core will cancel any outstanding reminders that a 
    match the ``ReminderCancelled`` event. For example,

    - ``ReminderCancelled(intent="greet")`` cancels all reminders with intent ``greet``
    - ``ReminderCancelled(entities={...})`` cancels all reminders with the given entities a 
    - ``ReminderCancelled("...")`` cancels the one unique reminder with the given name a 
    - ``ReminderCancelled()`` cancels all reminders a 


Pause a conversation a 
~~~~~~~~~~~~~~~~~~~~

:Short: Stops the bot from responding to messages. Action prediction a 
        will be halted until resumed.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER ConversationPaused a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.ConversationPaused a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: ConversationPaused.apply_to a 


Resume a conversation a 
~~~~~~~~~~~~~~~~~~~~~

:Short: Resumes a previously paused conversation. The bot will start a 
        predicting actions again.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER ConversationResumed a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.ConversationResumed a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: ConversationResumed.apply_to a 


Force a followup action a 
~~~~~~~~~~~~~~~~~~~~~~~

:Short: Instead of predicting the next action, force the next action a 
        to be a fixed one.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER FollowupAction a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.FollowupAction a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: FollowupAction.apply_to a 


Automatically tracked events a 
----------------------------


User sent message a 
~~~~~~~~~~~~~~~~~

:Short: Message a user sent to the bot.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :lines: 1-
      :start-after: # DOCS MARKER UserUttered a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.UserUttered a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: UserUttered.apply_to a 


Bot responded message a 
~~~~~~~~~~~~~~~~~~~~~

:Short: Message a bot sent to the user.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER BotUttered a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.BotUttered a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: BotUttered.apply_to a 


Undo a user message a 
~~~~~~~~~~~~~~~~~~~

:Short: Undoes all side effects that happened after the last user message a 
        (including the ``user`` event of the message).
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER UserUtteranceReverted a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.UserUtteranceReverted a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: UserUtteranceReverted.apply_to a 


Undo an action a 
~~~~~~~~~~~~~~

:Short: Undoes all side effects that happened after the last action a 
        (including the ``action`` event of the action).
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER ActionReverted a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.ActionReverted a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: ActionReverted.apply_to a 


Log an executed action a 
~~~~~~~~~~~~~~~~~~~~~~

:Short: Logs an action the bot executed to the conversation. Events that a 
        action created are logged separately.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER ActionExecuted a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.ActionExecuted a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: ActionExecuted.apply_to a 

Start a new conversation session a 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Marks the beginning of a new conversation session. Resets the tracker and a 
        triggers an ``ActionSessionStart`` which by default applies the existing a 
        ``SlotSet`` events to the new session.

:JSON:
    .. literalinclude:: ../../tests/core/test_events.py a 
      :start-after: # DOCS MARKER SessionStarted a 
      :dedent: 4 a 
      :end-before: # DOCS END a 
:Class:
    .. autoclass:: rasa.core.events.SessionStarted a 

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py a 
      :dedent: 4 a 
      :pyobject: SessionStarted.apply_to a 

