.. _slotfilling:

Slot Filling
============

One of the most common conversation patterns
is to collect a few pieces of information
from a user in order to do something (book a restaurant, call an API, search a database, etc.).
This is also called **slot filling**.



Example: Providing the Weather
------------------------------


Let's say you are building a weather bot ⛅️. If somebody asks you for the weather, you will
need to know their location. Users might say that right away, e.g. `What's the weather in Caracas?`
When they don't provide this information, you'll have to ask them for it. 
We can provide two stories to Rasa Core, so that it can learn to handle both cases:

.. code-block:: md

    # story1
    * ask_weather{"location": "Caracas"}
       - action_weather_api

    # story2
    * ask_weather
       - utter_ask_location
    * inform{"location": "Caracas"}
       - action_weather_api

Here we are assuming you have defined an ``inform`` intent, which captures the cases where a user 
is just providing information.

But :ref:`customactions` can also set slots, and these can also influence the conversation. 
For example, a location like `San Jose` could refer to multiple places, in this case, probably in
Costa Rica 🇨🇷  or California 🇺🇸

Let's add a call to a location API to deal with this. 
Start by defining a ``location_match`` slot:

.. code-block:: md
    
    slots:
      location_match:
        type: categorical
        values:
        - zero
        - one
        - multiple


And our location api action will have to use the API response to fill in this slot.
It can ``return [SlotSet("location_match", value)]``, where ``value`` is one of ``"zero"``, ``"one"``, or 
``"multiple"``, depending on what the API sends back. 

We then define stories for each of these cases:


.. code-block:: md
    :emphasize-lines: 12-13, 18-19, 24-25

    # story1
    * ask_weather{"location": "Caracas"}
       - action_location_api
       - slot{"location_match": "one"}
       - action_weather_api

    # story2
    * ask_weather
       - utter_ask_location
    * inform{"location": "Caracas"}
       - action_location_api
       - slot{"location_match": "one"}
       - action_weather_api

    # story3
    * ask_weather{"location": "the Moon"}
       - action_location_api
       - slot{"location_match": "none"}
       - utter_location_not_found

    # story4
    * ask_weather{"location": "San Jose"}
       - action_location_api
       - slot{"location_match": "multiple"}
       - utter_ask_which_location


Now we've given Rasa Core a few examples of how to handle the different values
that the ``location_match`` slot can take.
Right now, we still only have four stories, which is not a lot of training data.
:ref:`interactive_learning` is agreat way to explore more conversations 
that aren't in your stories already.
The best way to improve your model is to test it yourself, have other people test it,
and correct the mistakes it makes. 


Debugging
~~~~~~~~~

The first thing to try is to run your bot with the ``debug`` flag, see :ref:`debugging` for details.
If you are just getting started, you probably only have a few hand-written stories.
This is a great starting point, but 
you should give your bot to people to test **as soon as possible**. One of the guiding principles
behind Rasa Core is:

.. pull-quote:: Learning from real conversations is more important than designing hypothetical ones

So don't try to cover every possiblity in your hand-written stories before giving it to testers.
Real user behavior will always surprise you! 


Slot Filling with a ``FormAction``
----------------------------------

If you need to collect multiple pieces of information in a row, it is sometimes easier
to create a ``FormAction``.
This lets you have a single action that is called multiple times, rather than separate actions for each question,
e.g. ``utter_ask_cuisine``, ``utter_ask_numpeople``, in a restaurant bot. 


.. note::
    You don't *have* to use a ``FormAction`` to do slot filling! It just means you need 
    fewer stories to get the initial flow working. 

A form action has a set of required fields, which you define for the class:

.. literalinclude:: ../tests/test_forms.py
   :pyobject: ActionSearchRestaurants



The way this works is that every time you call this action, it will pick one of the 
``REQUIRED_FIELDS`` that's still missing and ask the user for it. You can also ask a yes/no
question with a ``BooleanFormField``.

The form action will set a slot called ``requested_slot`` to keep track if what it has asked the user.
So a story will look something like this:

.. code-block:: md

   * request_restaurant
        - action_restaurant_form
        - slot{"requested_slot": "people"}
   * inform{"number": 3}
        - action_restaurant_form
        - slot{"people": 3}
        - slot{"requested_slot": "cuisine"}
   * inform{"cuisine": "chinese"}
        - action_restaurant_form
        - slot{"cuisine": "chinese"}
        - slot{"requested_slot": "vegetarian"}
   * deny
        - action_restaurant_form
        - slot{"vegetarian": false}

Some important things to consider:

- The ``submit()`` method is called when the action is run and all slots are filled, in this case after the ``deny`` intent.
  If you are just collecting some information and don't need to make an API call at the end, your ``submit()`` method
  should just ``return []``.
- Your domain needs to have a slot called ``requested_slot``. You can make this an unfeaturized slot.
- You need to define utterances for asking for each slot in your domain, e.g. ``utter_ask_{slot_name}``.
- We strongly recommend that you create these stories using interactive learning, because if you
  type these by hand you will probably forget to include the lines for the ``requested_slot``.
- Any slots that are already set won't be asked for. E.g. if someone says "I'd like a vegetarian Chinese restaurant for 8 people" the ``submit`` function should get called right away.


Form Fields and Free-text Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pre-defined ``FormField`` types are:

- ``EntityFormField(entity_name, slot_name)``, which will
  look for an entity called ``entity_name`` to fill a slot ``slot_name``.
- ``BooleanFormField(slot_name, affirm_intent, deny_intent)``, which looks for the intents ``affirm_intent``
  and ``deny_intent`` to fill a boolean slot called ``slot_name``.
- ``FreeTextFormField(slot_name)``, which will use the next user utterance to fill the text slot ``slot_name``.

For any subclass of  ``FormField``, its ``validate()`` method will be called before setting it 
as a slot. By default this just checks that the value isn't ``None``, but if you want to check 
the value against a DB, or check a pattern is matched, you can do so by defining your own class
like ``MyCustomFormField`` and overriding the ``validate()`` method.

.. warning:: 

   The ``FreeTextFormField`` class will just extract the user message as a value.
   However, there is currently no way to write a 'wildcard' intent in Rasa Core stories as of now. 
   Typically your NLU model will assign this free-text input to 2-3 different intents. 
   It's easiest to add stories for each of these. 

