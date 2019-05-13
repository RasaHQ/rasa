:desc: Use events in open source library Rasa Core to support functionalities
       like resetting slots, scheduling reminder or pausing a conversation.

.. _events:

Events
======
List of all the events the dialogue system is able to handle and supports.

Most of the time, you will set or receive events in their json
representation. E.g. if you are modifying a conversation over the
HTTP API, or if you are implementing your own action server.

.. contents::

General purpose events
----------------------

Event Base Class
~~~~~~~~~~~~~~~~


:Short: Base class for all of Rasa Core's events. **Can not directly be used!**
:Description:
    Contains common functionality for all events. E.g. it ensures, that
    every event has a timestamp and a ``event`` attribute that describes
    the type of the event (e.g. ``slot`` for the slot setting event).
:Class:
    .. autoclass:: rasa.core.events.Event

Set a Slot
~~~~~~~~~~

:Short: Event to set a slot on a tracker
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER SetSlot
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.SlotSet

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: SlotSet.apply_to


Restart a conversation
~~~~~~~~~~~~~~~~~~~~~~

:Short: Resets anything logged on the tracker.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER Restarted
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.Restarted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: Restarted.apply_to


Reset all Slots
~~~~~~~~~~~~~~~

:Short: Resets all the slots of a conversation.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER AllSlotsReset
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.AllSlotsReset

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: AllSlotsReset.apply_to


Schedule a reminder
~~~~~~~~~~~~~~~~~~~

:Short: Schedule an action to be executed in the future.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER ReminderScheduled
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ReminderScheduled

:Effect:
    When added to a tracker, core will schedule the action to be
    run in the future.

Pause a conversation
~~~~~~~~~~~~~~~~~~~~

:Short: Stops the bot from responding to messages. Action prediction
        will be halted until resumed.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER ConversationPaused
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ConversationPaused

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: ConversationPaused.apply_to


Resume a conversation
~~~~~~~~~~~~~~~~~~~~~

:Short: Resumes a previously paused conversation. The bot will start
        predicting actions again.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER ConversationResumed
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ConversationResumed

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: ConversationResumed.apply_to


Force a followup action
~~~~~~~~~~~~~~~~~~~~~~~

:Short: Instead of predicting the next action, force the next action
        to be a fixed one.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER FollowupAction
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.FollowupAction

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: FollowupAction.apply_to


Automatically tracked events
----------------------------


User sent message
~~~~~~~~~~~~~~~~~

:Short: Message a user sent to the bot.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER UserUttered
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.UserUttered

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: UserUttered.apply_to


Bot responded message
~~~~~~~~~~~~~~~~~~~~~

:Short: Message a bot sent to the user.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER BotUttered
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.BotUttered

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: BotUttered.apply_to


Undo a user message
~~~~~~~~~~~~~~~~~~~

:Short: Undoes all side effects that happened after the last user message
        (including the ``user`` event of the message).
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER UserUtteranceReverted
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.UserUtteranceReverted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: UserUtteranceReverted.apply_to


Undo an action
~~~~~~~~~~~~~~

:Short: Undoes all side effects that happened after the last action
        (including the ``action`` event of the action).
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER ActionReverted
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ActionReverted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: ActionReverted.apply_to


Log an executed action
~~~~~~~~~~~~~~~~~~~~~~

:Short: Logs an action the bot executed to the conversation. Events that
        action created are logged separately.
:JSON:
    .. literalinclude:: ../../../tests/core/test_events.py
      :lines: 1-
      :dedent: 4
      :start-after: # DOCS MARKER ActionExecuted
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ActionExecuted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../../rasa/core/events/__init__.py
      :pyobject: ActionExecuted.apply_to


.. include:: ../feedback.inc
