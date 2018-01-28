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
information again, but the *value* of a ``TextSlot`` has no impact on which actions Rasa Core predicts. This is
explained in :ref:`more detail below <slot_features>`.

The full set of slot types and their behaviour is described here: :ref:`slot_types`.

Using Slot Values to Influence Which Actions are Predicted
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 

**Custom Slots**

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
   * inform{"people": "3"}
   - action_book_table
   ...
   # story2
   * inform{"people": "9"}
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

**Slot features**

.. _slot_features:

When Rasa Core runs training against your story, the presence of a ``Slot`` entry will be used to help 
determine the next action that should be taken. 

This works best with ``CategoricalSlot`` slot types. A ``TextSlot`` can have any value, but it only has one 
feature - set ``(1)`` or not set ``(0)``. A ``CategoricalSlot`` has a set number of values and a feature for each. Rasa core 
will be able to make decisions based not only on whether the value is set but also on the value itself.

.. code-block:: yaml

    restaurant_availability:
        type: categorical
        values:
        - unknown
        - booked-out
        - waiting-list
        - available

When the ``restaurant_availability`` slot is set Rasa Core will be able to determine if the restaurant in question is
available and choose radically different actions to perform based on the value.

The ``Slot`` *might* be set by Rasa Core itself from entities detected by the NLU module, but usually you would 
return the value of the ``Slot`` from your ``Action`` and then use the next ``Turn`` of the conversation to
check what feature is set.

.. code-block:: python

    def run(self, dispatcher, tracker, domain):
        # some logic here to decide if the restaurant is "available", "booked-out" or whatever
        return [SlotSet("restaurant_availability", "available")]

In this first story we will try and make a booking for 5 people in a restaurant on the night of 21st August 2018. 
In this case the restaurant is booked out so we want to apologise to the customer and suggests similar restaurants.

**Note:** it is assumed that the Rasa Core model has been trained to recognise a message like *"Book Murphys Bistro on August 21 for 5 people"* 

.. code-block:: md

    # restaurant unavailable
    * _make_booking[people=5, date="2018-08-21T19:30:00+00:00", restaurant_id=145]
    - slot{"restaurant_availability": "booked-out"}
    - utter_sorry_unavailable
    - action_show_similar

This second story details the flow when the restaurant is available. We will tell the customer we have booked 
the restaurant and ask if any further help is required.
    
.. code-block:: md

    # restaurant available
    * _make_booking[people=5, date="2018-08-22T19:30:00+00:00", restaurant_id=145]
    - slot{"restaurant_availability": "available"}
    - action_make_booking
    - utter_restaurant_booked
    - utter_anything_more
    * _bye
    - utter_thank_you

In this last example, the intent ``make_booking`` was found but either Rasa Core failed to parse a date or the 
date was not provided. In this case we would need to ask for more information.

**Note:** this last story is using the fact that ``date`` is a ``TextSlot`` and therefore has a single feature that 
is set or not.

.. code-block:: md

    # restaurant request without date
    * _make_booking[people=5, restaurant_id=145]
    - slot{"date": null}
    - utter_date_required
    * _inform[date="2018-08-22T19:30:00+00:00"]
    - action_make_booking
    - utter_restaurant_booked
    - utter_anything_more
    * _bye
    - utter_thank_you


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
