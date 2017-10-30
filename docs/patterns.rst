.. _patterns:

Common Patterns
===============


.. note:: 
   Here we go through some typical conversation patterns and explain how to implement
   them with Rasa Core. 


Rasa Core uses ML models to predict actions, and ``slots`` provide important
information these models rely on to keep track of context and handle multi-turn dialogue.
Slots are for storing information that's relevant over multiple turns. For example, in
our restaurant example, we would want to keep track of things like the cuisine and number of 
people for the duration of the conversation. 

Collecting Information to Complete a Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For collecting a set of preferences, you can use ``TextSlot`` s like in the restaurant example:

.. code-block:: yaml

   slots:
     cuisine:
       type: text
    people:
       type: text
    ...


When Rasa sees an entity with the same name as one of the slots, this value is automatically saved.
For example, if your NLU module detects the entity ``people=8`` in the sentence *"I'd like a table for 8"*,
this will be saved as a slot,

.. testsetup::

   from rasa_core.trackers import DialogueStateTracker
   from rasa_core.slots import TextSlot
   from rasa_core.events import SlotSet
   tracker = DialogueStateTracker("default", slots=[TextSlot("people")])
   tracker.update(SlotSet("people", "8"))

.. doctest::

   >>> tracker.slots
   {'people': <TextSlot(people: 8)>}


When Rasa Core predicts the next action to take, the only information it has about the ``TextSlot`` s is 
**whether or not they are defined**. So you have enough information to know that you don't have to ask for this
information again, but the *value* has no impact on which actions Rasa Core predicts.
The full set of slot types and their behaviour is described here: :ref:`slot_types`.

Using Slot Values to Influence Which Actions are Predicted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Maybe your restaurant booking system can only handle bookings for up to 6 people, so the 
request above isn't valid. In this case you want the *value* of the slot to influence the 
next selected action (not just whether it's been specified). You can achieve this using a custom
slot class. 

The way we defined it below, if the number of people is less than or equal to 6, we return ``(1,0)``,
if it's more we return ``(0,1)``, and if it's not set ``(0,0)``. 

Rasa Core can use that information to distinguish between different situations - so long as 
you have some training stories where the appropriate responses take place, e.g.:


.. code-block:: md

   # story1
   ...
   * _inform[people=3]
   - action_book_table
   ...
   # story2
   * _inform[people=9]
   - action_explain_table_limit
   


.. doctest::

   from rasa_core.slots import Slot
   
   class NumberOfPeopleSlot(Slot):
     
     def feature_dimensionality(self):
         return 2
    
     def as_feature(self):
         r = [0.0] * self.feature_dimensionality()
         if self.value:
             if self.value <= 6:
                 r[0] = 1.0
             else:
                 r[1] = 1.0
         return r


If you want to store something like the price range, this is actually a little simpler. Variables
like price range usually take on one-of-n values, e.g. low, medium, high. For these cases you can use
a ``categorical`` slot.

.. code-block:: yaml

   slots:
     price_range:
       type: categorical
       values: low, medium, high


Rasa automatically represents (featurises) this as a one-hot encoding of the values: ``(1,0,0)``, ``(0,1,0)``, or ``(0,0,1)``.


Storing API responses in the tracker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You often want to save some information in the tracker, like the results from 
a database query, or from an API call. If you don't want the value to influence the 
dialogue, you can use a ``unfeaturized`` slot. You can explicitly set this value in a custom action:

.. doctest::

   from rasa_core.actions import Action
   from rasa_core.events import SlotSet
   import requests
   
   class ApiAction(Action):
       def name(self):
           return "api_action"

       def run(self, tracker, dispatcher):
           data = requests.get(url).json
           return [SlotSet("api_result", data)]
