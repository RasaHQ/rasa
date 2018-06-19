.. _slotfilling:

Slot Filling
============

One of the most common conversation patterns
is to collect a few pieces of information
from a user in order to do something (book a restaurant, call an API, search a database, etc.).
This is also called **slot filling**.



Slot Filling with Regular Actions
---------------------------------

The `bAbI restaurant task <https://github.com/RasaHQ/rasa_core/tree/master/examples/restaurantbot>`_
is an example of a bot that learns to ask the questions it needs in order to recommend a restaurant.
Once the user has provided the location, price, cuisine, and number of people, the bot is ready to start
searching.
Your users will of course not provide all the information in the same order, so you will need multiple
stories to cover the different ways people can provide information. Here is one example:

.. code-block:: md

    ## story_07715946
    * greet
       - action_ask_howcanhelp
    * inform{"location": "rome", "price": "cheap"}
       - action_on_it
       - action_ask_cuisine
    * inform{"cuisine": "spanish"}
       - action_ask_numpeople
    * inform{"people": "six"}
       - action_ack_dosearch


Slot Filling with a ``FormAction``
----------------------------------

To make this easier, Rasa has a special action class called ``FormAction``.
This lets you have a single action for this task, rather than separate actions for each question,
e.g. ``utter_ask_cuisine``, ``utter_ask_numpeople``, etc. 


.. note::
    You don't *have* to use a ``FormAction`` to do slot filling! It just means you need 
    fewer stories to get the initial flow working. 

A form action has a set of required fields, which you define for the class:

.. literalinclude:: ../tests/test_forms.py
   :pyobject: ActionSearchRestaurants



The way this works is that every time you call this action, it will pick one of the 
``REQUIRED_FIELDS`` that's still missing and ask the user for it. You can also ask a yes/no
question with a ``BooleanFormField``.

The story will look something like this:

.. code-block:: md

   * request_restaurant
        - action_restaurant_form
        - slot{"requested_slot": "people"}
   * inform{"number": 3}
        - action_restaurant_form
        - slot{"people": 3}
        - slot{"requested_slot": "time"}
   * inform{"time": "8pm"}
      - action_restaurant_form


Some important things to consider:

- Your domain needs to have a slot called ``requested_slot``. This can be unfeaturized, but if you want
  to support contextual questions like *"why do you need to know that information?"* it will help if you make this
  a categorical slot. 
- You need to define utterances for asking for each slot in your domain, e.g. ``utter_ask_{slot_name}``.
- We strongly recommend that you create these stories using interactive learning, because if you
  type these by hand you will probably forget to add all of the slots.
- Any slots that are already set won't be asked for. E.g. if someone says "I'd like a Chinese restaurant for 8 people" the ``submit`` function should get called right away.


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

