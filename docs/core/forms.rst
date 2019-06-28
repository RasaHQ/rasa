:desc: Follow a rule-based process of information gathering using FormActions
       in open source bot framework Rasa.

.. _forms:

Forms
=====

.. note::
   There is an in-depth tutorial `here <https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/>`_ about how to use Rasa Forms for slot filling.

.. contents::
   :local:

One of the most common conversation patterns is to collect a few pieces of
information from a user in order to do something (book a restaurant, call an
API, search a database, etc.). This is also called **slot filling**.


If you need to collect multiple pieces of information in a row, we recommended
that you create a ``FormAction``. This is a single action which contains the
logic to loop over the required slots and ask the user for this information.
There is a full example using forms in the ``examples/formbot`` directory of
Rasa Core.


When you define a form, you need to add it to your domain file.
If your form's name is ``restaurant_form``, your domain would look like this:

.. code-block:: yaml

   forms:
     - restaurant_form
   actions:
     ...

See ``examples/formbot/domain.yml`` for an example.

Configuration File
------------------

To use forms, you also need to include the ``FormPolicy`` in your policy
configuration file. For example:

.. code-block:: yaml

  policies:
    - name: "FormPolicy"

see ``examples/formbot/domain.yml`` for an example.

Form Basics
-----------

Using a ``FormAction``, you can describe *all* of the happy paths with a single
story. By "happy path", we mean that whenever you ask a user for some information,
they respond with the information you asked for.

If we take the example of the restaurant bot, this single story describes all of the
happy paths.

.. code-block:: story

    ## happy path
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - form{"name": null}

In this story the user intent is ``request_restaurant``, which is followed by
the form action ``restaurant_form``. With ``form{"name": "restaurant_form"}`` the
form is activated and with ``form{"name": null}`` the form is deactivated again.
As shown in the section :ref:`section_unhappy` the bot can execute any kind of
actions outside the form while the form is still active. On the "happy path",
where the user is cooperating well and the system understands the user input correctly,
the form is filling all requested slots without interruption.

The ``FormAction`` will only request slots which haven't already been set.
If a user starts the conversation with
`I'd like a vegetarian Chinese restaurant for 8 people`, then they won't be
asked about the ``cuisine`` and ``num_people`` slots.

Note that for this story to work, your slots should be :ref:`unfeaturized
<unfeaturized-slot>`. If any of these slots are featurized, your story needs to
include ``slot{}`` events to show these slots being set. In that case, the
easiest way to create valid stories is to use :ref:`interactive-learning`.

In the story above, ``restaurant_form`` is the name of our form action.
Here is an example of what it looks like.
You need to define three methods:

- ``name``: the name of this action
- ``required_slots``: a list of slots that need to be filled for the ``submit`` method to work.
- ``submit``: what to do at the end of the form, when all the slots have been filled.

.. literalinclude:: ../../examples/formbot/actions.py
   :dedent: 4
   :pyobject: RestaurantForm.name

.. literalinclude:: ../../examples/formbot/actions.py
   :dedent: 4
   :pyobject: RestaurantForm.required_slots

.. literalinclude:: ../../examples/formbot/actions.py
   :dedent: 4
   :pyobject: RestaurantForm.submit

Once the form action gets called for the first time,
the form gets activated and the ``FormPolicy`` jumps in.
The ``FormPolicy`` is extremely simple and just always predicts the form action.
See :ref:`section_unhappy` for how to work with unexpected user input.

Every time the form action gets called, it will ask the user for the next slot in
``required_slots`` which is not already set.
It does this by looking for a template called ``utter_ask_{slot_name}``,
so you need to define these in your domain file for each required slot.

Once all the slots are filled, the ``submit()`` method is called, where you can
use the information you've collected to do something for the user, for example
querying a restaurant API.
If you don't want your form to do anything at the end, just use ``return []``
as your submit method.
After the submit method is called, the form is deactivated,
and other policies in your Core model will be used to predict the next action.

Custom slot mappings
--------------------

If you do not define slot mappings, slots will be only filled by entities
with the same name as the slot that are picked up from the user input.
Some slots, like ``cuisine``, can be picked up using a single entity, but a
``FormAction`` can also support yes/no questions and free-text input.
The ``slot_mappings`` method defines how to extract slot values from user responses.

Here's an example for the restaurant bot:

.. literalinclude:: ../../examples/formbot/actions.py
   :dedent: 4
   :pyobject: RestaurantForm.slot_mappings

The predefined functions work as follows:

- ``self.from_entity(entity=entity_name, intent=intent_name)``
  will look for an entity called ``entity_name`` to fill a slot
  ``slot_name`` regardless of user intent if ``intent_name`` is ``None``
  else only if the users intent is ``intent_name``.
- ``self.from_intent(intent=intent_name, value=value)``
  will fill slot ``slot_name`` with ``value`` if user intent is ``intent_name``.
  To make a boolean slot, take a look at the definition of ``outdoor_seating``
  above. Note: Slot will not be filled with user intent of message triggering
  the form action. Use ``self.from_trigger_intent`` below.
- ``self.from_trigger_intent(intent=intent_name, value=value)``
  will fill slot ``slot_name`` with ``value`` if form was triggered with user
  intent ``intent_name``.
- ``self.from_text(intent=intent_name)`` will use the next
  user utterance to fill the text slot ``slot_name`` regardless of user intent
  if ``intent_name`` is ``None`` else only if user intent is ``intent_name``.
- If you want to allow a combination of these, provide them as a list as in the
  example above


Validating user input
---------------------

After extracting a slot value from user input, the form will try to validate the
value of the slot. Note that by default, validation only happens if the form
action is executed immediately after user input. This can be changed in the
``_validate_if_required()`` function of the ``FormAction`` class in Rasa SDK.
Any required slots that were filled before the initial activation of a form
are validated upon activation as well.

By default, validation only checks if the requested slot was successfully
extracted from the slot mappings. If you want to add custom validation, for
example to check a value against a database, you can do this by writing a helper
validation function with the name ``validate_{slot-name}``.

Here is an example , ``validate_cuisine()``, which checks if the extracted cuisine slot
belongs to a list of supported cuisines.

.. literalinclude:: ../../examples/formbot/actions.py
   :pyobject: RestaurantForm.cuisine_db

.. literalinclude:: ../../examples/formbot/actions.py
   :pyobject: RestaurantForm.validate_cuisine

As the helper validation functions return dictionaries of slot names and values
to set, you can set more slots than just the one you are validating from inside
a helper validation method. However, you are responsible for making sure that
those extra slot values are valid.

You can also deactivate the form directly during this validation step (in case the
slot is filled with something that you are certain can't be handled) by returning
``self.deactivate()``

If nothing is extracted from the user's utterance for any of the required slots, an
``ActionExecutionRejection`` error will be raised, meaning the action execution
was rejected and therefore Core will fall back onto a different policy to
predict another action.

.. _section_unhappy:

Handling unhappy paths
----------------------

Of course your users will not always respond with the information you ask of them.
Typically, users will ask questions, make chitchat, change their mind, or otherwise
stray from the happy path. The way this works with forms is that a form will raise
an ``ActionExecutionRejection`` if the user didn't provide the requested information.
You need to handle events that might cause ``ActionExecutionRejection`` errors
in your stories. For example, if you expect your users to chitchat with your bot,
you could add a story like this:

.. code-block:: story

    ## chitchat
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
    * chitchat
        - utter_chitchat
        - restaurant_form
        - form{"name": null}

In some situations, users may change their mind in the middle of form action
and decide not to go forward with their initial request. In cases like this, the
assistant should stop asking for the requested slots. You can handle such situations
gracefully using a default action ``action_deactivate_form`` which will deactivate
the form and reset the requested slot. An example story of such conversation could
look as follows:

.. code-block:: story

    ## chitchat
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
    * stop
        - utter_ask_continue
    * deny
        - action_deactivate_form
        - form{"name": null}


It is **strongly** recommended that you build these stories using interactive learning.
If you write these stories by hand you will likely miss important things.
Please read :ref:`section_interactive_learning_forms`
on how to use interactive learning with forms.

The requested_slot slot
-----------------------

The slot ``requested_slot`` is automatically added to the domain as an
unfeaturized slot. If you want to make it featurized, you need to add it
to your domain file as a categorical slot. You might want to do this if you
want to handle your unhappy paths differently depending on what slot is
currently being asked from the user. For example, say your users respond
to one of the bot's questions with another question, like *why do you need to know that?*
The response to this ``explain`` intent depends on where we are in the story.
In the restaurant case, your stories would look something like this:

.. code-block:: story

    ## explain cuisine slot
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - slot{"requested_slot": "cuisine"}
    * explain
        - utter_explain_cuisine
        - restaurant_form
        - slot{"cuisine": "greek"}
        ( ... all other slots the form set ... )
        - form{"name": null}

    ## explain num_people slot
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - slot{"requested_slot": "num_people"}
    * explain
        - utter_explain_num_people
        - restaurant_form
        - slot{"cuisine": "greek"}
        ( ... all other slots the form set ... )
        - form{"name": null}

Again, is is **strongly** recommended that you use interactive
learning to build these stories.
Please read :ref:`section_interactive_learning_forms`
on how to use interactive learning with forms.

.. _conditional-logic:

Handling conditional slot logic
-------------------------------

Many forms require more logic than just requesting a list of fields.
For example, if someone requests ``greek`` as their cuisine, you may want to
ask if they are looking for somewhere with outside seating.

You can achieve this by writing some logic into the ``required_slots()`` method,
for example:

.. code-block:: python

    @staticmethod
    def required_slots(tracker) -> List[Text]:
       """A list of required slots that the form has to fill"""

       if tracker.get_slot('cuisine') == 'greek':
         return ["cuisine", "num_people", "outdoor_seating",
                 "preferences", "feedback"]
       else:
         return ["cuisine", "num_people",
                 "preferences", "feedback"]

This mechanism is quite general and you can use it to build many different
kinds of logic into your forms.



Debugging
---------

The first thing to try is to run your bot with the ``debug`` flag, see :ref:`command-line-interface` for details.
If you are just getting started, you probably only have a few hand-written stories.
This is a great starting point, but
you should give your bot to people to test **as soon as possible**. One of the guiding principles
behind Rasa Core is:

.. pull-quote:: Learning from real conversations is more important than designing hypothetical ones

So don't try to cover every possibility in your hand-written stories before giving it to testers.
Real user behavior will always surprise you!
