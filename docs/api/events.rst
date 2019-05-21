:desc: Use events in open source library Rasa Core to support functionalities
       like resetting slots, scheduling reminder or pausing a conversation.

.. _events:

Events
======

Conversations in Rasa are represented as a sequence of events.
This page lists the event types defined in Rasa Core.

.. note::
    If you are using the Rasa SDK to write custom actions in python,
    you need to import the events from ``rasa_sdk.events``, not from
    ``rasa.core.events``. If you are writing actions in another language,
    your events should be formatted like the JSON objects on this page.



.. contents::
   :local:

General Purpose Events
----------------------

Set a Slot
~~~~~~~~~~

:Short: Event to set a slot on a tracker
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER SetSlot
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.SlotSet

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: SlotSet.apply_to


Restart a conversation
~~~~~~~~~~~~~~~~~~~~~~

:Short: Resets anything logged on the tracker.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER Restarted
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.Restarted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: Restarted.apply_to


Reset all Slots
~~~~~~~~~~~~~~~

:Short: Resets all the slots of a conversation.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER AllSlotsReset
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.AllSlotsReset

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: AllSlotsReset.apply_to


Schedule a reminder
~~~~~~~~~~~~~~~~~~~

:Short: Schedule an action to be executed in the future.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :lines: 1-
      :start-after: # DOCS MARKER ReminderScheduled
      :dedent: 4
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
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER ConversationPaused
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ConversationPaused

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: ConversationPaused.apply_to


Resume a conversation
~~~~~~~~~~~~~~~~~~~~~

:Short: Resumes a previously paused conversation. The bot will start
        predicting actions again.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER ConversationResumed
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ConversationResumed

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: ConversationResumed.apply_to


Force a followup action
~~~~~~~~~~~~~~~~~~~~~~~

:Short: Instead of predicting the next action, force the next action
        to be a fixed one.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER FollowupAction
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.FollowupAction

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: FollowupAction.apply_to


Automatically tracked events
----------------------------


User sent message
~~~~~~~~~~~~~~~~~

:Short: Message a user sent to the bot.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :lines: 1-
      :start-after: # DOCS MARKER UserUttered
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.UserUttered

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: UserUttered.apply_to


Bot responded message
~~~~~~~~~~~~~~~~~~~~~

:Short: Message a bot sent to the user.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER BotUttered
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.BotUttered

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: BotUttered.apply_to


Undo a user message
~~~~~~~~~~~~~~~~~~~

:Short: Undoes all side effects that happened after the last user message
        (including the ``user`` event of the message).
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER UserUtteranceReverted
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.UserUtteranceReverted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: UserUtteranceReverted.apply_to


Undo an action
~~~~~~~~~~~~~~

:Short: Undoes all side effects that happened after the last action
        (including the ``action`` event of the action).
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER ActionReverted
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ActionReverted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: ActionReverted.apply_to


Log an executed action
~~~~~~~~~~~~~~~~~~~~~~

:Short: Logs an action the bot executed to the conversation. Events that
        action created are logged separately.
:JSON:
    .. literalinclude:: ../../tests/core/test_events.py
      :start-after: # DOCS MARKER ActionExecuted
      :dedent: 4
      :end-before: # DOCS END
:Class:
    .. autoclass:: rasa.core.events.ActionExecuted

:Effect:
    When added to a tracker, this is the code used to update the tracker:

    .. literalinclude:: ../../rasa/core/events/__init__.py
      :dedent: 4
      :pyobject: ActionExecuted.apply_to
