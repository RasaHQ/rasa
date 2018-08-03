Events
======
List of all the events the dialogue system is able to handle and supports.

An event can be referenced by its ``type_name`` attribute (e.g. when returning
events in an http callback).

.. contents::


The Event base class
--------------------

.. autoclass:: rasa_core.events.Event


Events for Actions to Return
----------------------------


SlotSet
^^^^^^^

.. autoclass:: rasa_core.events.SlotSet

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: SlotSet.apply_to


Restarted
^^^^^^^^^

.. autoclass:: rasa_core.events.Restarted


.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: Restarted.apply_to



AllSlotsReset
^^^^^^^^^^^^^

.. autoclass:: rasa_core.events.AllSlotsReset

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: AllSlotsReset.apply_to


ReminderScheduled
^^^^^^^^^^^^^^^^^

.. autoclass:: rasa_core.events.ReminderScheduled


ConversationPaused
^^^^^^^^^^^^^^^^^^

.. autoclass:: rasa_core.events.ConversationPaused

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: ConversationPaused.apply_to


ConversationResumed
^^^^^^^^^^^^^^^^^^^

.. autoclass:: rasa_core.events.ConversationResumed

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: ConversationResumed.apply_to



Automatically tracked events
----------------------------

UserUttered
^^^^^^^^^^^

.. autoclass:: rasa_core.events.UserUttered

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: UserUttered.apply_to


BotUttered
^^^^^^^^^^

.. autoclass:: rasa_core.events.BotUttered

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: BotUttered.apply_to



UserutteranceReverted
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: rasa_core.events.UserUtteranceReverted

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: UserUtteranceReverted.apply_to



ActionReverted
^^^^^^^^^^^^^^

.. autoclass:: rasa_core.events.ActionReverted

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: ActionReverted.apply_to



ActionExecuted
^^^^^^^^^^^^^^

.. autoclass:: rasa_core.events.ActionExecuted

.. literalinclude:: ../../rasa_core/events/__init__.py
   :pyobject: ActionExecuted.apply_to

