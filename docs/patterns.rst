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

.. _slot_features:

**Slot features**

When Rasa Core runs trains a dialogue model using your stories the presence of a ``Slot`` entry will be used to help 
determine the next action that should be taken. This works best with ``CategoricalSlot`` slot types. 

A ``TextSlot`` can have any value, but it only has a single feature. It can be set, in which case the feature 
has a value ``(1)``, or if it is not set it will have a value ``(0)``. 

A ``CategoricalSlot`` has a number of values each of which is a feature. Taking the example below, when the 
``restaurant_availability`` slot is set Rasa Core will be able to determine whether or not the restaurant in question is 
available and choose radically different actions to perform based on the value.

.. code-block:: yaml

    restaurant_availability:
        type: categorical
        values:
        - unknown
        - booked-out
        - waiting-list
        - available

A ``Slot`` will be set by Rasa Core if its name and the name of the entity detected by the NLU module 
match. The value of the slot will influence the story dialogue if you add the slot to the training 
stories - this is explained in the examples below. Slots can also be set explicitly from our own custom ``Action`` 
and influence the dialogue based on real-world information.

.. code-block:: python

    class ActionMakeBooking(Action):

        def run(self, dispatcher, tracker, domain):
            restaurant_name=tracker.get_slot("restaurant_name")
            location=tracker.get_slot("location")
            num_people=tracker.get_slot("people")
            date=tracker.get_slot("date")
            # this will fetch the availability of the restaurant from your DB or an API
            availability=restaurantService.check_availability(restaurant_name, location, num_people, date)
            return [SlotSet("restaurant_availability", availability)]

The snippet of code above from a hypothetical ``Action`` shows that the value of the slot ``restaurant_availability`` is determined by querying a database or API. The restaurant availability is not something that is known when we train the dialogue model, the ``Slot`` value is the only way we can alter the course of the conversation based on information from the outside world.

The data fetched from an API call can also be stored for later use without altering the outcome of a conversation as detailed in :ref:`unfeaturized_slots`.

**Slot Features Example**

.. note:: 
    These example stories have been constructed manually for illustrative purposes. While this is a valid approach to training your model the preferred approach is to use :ref:`interactive learning <tutorial_interactive_learning>` which generates stories that are *much* less error-prone.

In this first story we will try and make a booking for 5 people in a restaurant on the night of 21st August 2018. 
In this case the restaurant is booked out so we want to apologize to the customer and suggest similar restaurants. It is assumed that the Rasa Core model has been trained to recognise a message like *"Book Murphys Bistro on August 21 for 5 people"*

.. code-block:: md

    # restaurant unavailable
    * _make_booking{"people":"5", "date":"2018-08-21T19:30:00+00:00", "restaurant_id":"145"}
    - slot{"restaurant_availability": "booked-out"}
    - utter_sorry_unavailable
    - action_show_similar

This second story details the flow when the restaurant is available. We will tell the customer we have booked 
the restaurant and ask if any further help is required.
    
.. code-block:: md
    # restaurant available
    * _make_booking{"people":"5", "date":"2018-08-22T19:30:00+00:00", "restaurant_id":"145"}
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
    * _make_booking{"people":"5", "restaurant_id":"145"}
    - slot{"date": null}
    - utter_date_required
    * _inform{"date":"2018-08-22T19:30:00+00:00"}
    - action_make_booking
    - utter_restaurant_booked
    - utter_anything_more
    * _bye
    - utter_thank_you

.. _unfeaturized_slots:

Storing API responses in the tracker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The result from an API call can be stored in a ``Slot`` as :ref:`explained above <slot_features>`. In that case 
the data is stored in a ``Slot`` that is featurized, influencing the flow of the dialogue. 

A slot of type ``unfeaturized`` can be used to store the results from a database query or API call so that it will 
not influence the course of a dialogue. An exaple ``unfeaturized`` slot defined in a domain file:

.. code-block:: yaml

    slots:
        api_result:
            type: unfeaturized

You can set this value in a custom ``Action``:

.. code-block:: python

   from rasa_core.actions import Action
   from rasa_core.events import SlotSet
   import requests
   
   class ApiAction(Action):
       def name(self):
           return "api_action"

       def run(self, tracker, dispatcher):
           data = requests.get(url).json
           return [SlotSet("api_result", data)]

This is especially useful when you are :ref:`persisting your tracker <persisting_trackers>` in Redis or another data store. You could cache the API or database responses separately, but storing them in the tracker means they will be persisted automatically with the rest of the dialogue state, and will be restored along with the rest of the state should the system require a reboot.