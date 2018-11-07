.. _slotfilling:

Slot Filling
============

.. contents::

One of the most common conversation patterns is to collect a few pieces of
information from a user in order to do something (book a restaurant, call an
API, search a database, etc.). This is also called **slot filling**.


If you need to collect multiple pieces of information in a row, we recommended 
that you create a ``FormAction``. This is a single action which contains the 
logic to loop over the required slots and ask the user for this information.
There is a full example using forms in the ``examples/formbot`` directory of 
Rasa Core.

You can take a look at the FormAction base class by clicking this link:

.. autoclass:: rasa_core_sdk.forms.FormAction

.. _section_form_basics:

Basics
------

Using a ``FormAction``, you can describe *all* of the happy paths with a single 
story. By "happy path", we mean that whenever you ask a user for some information, 
they respond with what you asked for.

If we take the example of the restaurant bot, this single story describes all of the 
happy paths. 

.. code-block:: story

    ## happy path
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - form{"name": null}

The ``FormAction`` will only requests slots which haven't already been set.
If a user says 
"I'd like a vegetarian Chinese restaurant for 8 people", they won't be 
asked about the ``cuisine`` and ``num_people`` slots.

Note that for this story to work, your slots should be `unfeaturized
<https://rasa.com/docs/core/api/slots_api/#unfeaturized-slot>`_.
If they're not, you should add all the slots that have been set by the form.

The ``restaurant_form`` in the story above is the name of our form action.
Here is an example of what it looks like. 
You need to define three methods:

- ``name``: the name of this action
- ``required_slots``: a list of slots that need to be filled for the ``submit`` method to work.
- ``submit``: what to do at the end of the form, when all the slots have been filled.


.. code-block:: python

 class RestaurantForm(FormAction):
    """Example of a custom form action"""

    def name(self):
        # type: () -> Text
        """Unique identifier of the form"""

        return "restaurant_form"

    @staticmethod
    def required_slots(tracker):
        # type: () -> List[Text]
        """A list of required slots that the form has to fill"""

        return ["cuisine", "num_people", "outdoor_seating",
                "preferences", "feedback"]

    def submit(self, dispatcher, tracker, domain):
        # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
        """Define what the form has to do
            after all required slots are filled"""

        # utter submit template
        dispatcher.utter_template('utter_submit', tracker)
        return []


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

Some slots (like ``cuisine``) can be picked up using a single entity, but a 
``FormAction`` can also support yes/no questions and free-text input.
The ``slot_mapping`` method defines how to extract slot values from user responses.

Here's an example for the restaurant bot:

.. code-block:: python

    def slot_mapping(self):
        # type: () -> Dict[Text: Union[Text, Dict, List[Text, Dict]]]
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where the first match will be picked"""

        return {"cuisine": self.from_entity(entity="cuisine",
                                            intent="inform"),
                "num_people": self.from_entity(entity="number"),
                "outdoor_seating": [self.from_entity(entity="seating"),
                                    self.from_intent(intent='affirm',
                                                     value=True),
                                    self.from_intent(intent='deny',
                                                     value=False)],
                "preferences": [self.from_text(intent='inform'),
                                self.from_intent(intent='deny',
                                                 value="no additional "
                                                       "preferences")],
                "feedback": [self.from_entity(entity="feedback"),
                             self.from_text()]}

The predefined functions work as follows:

- ``self.from_entity(entity=entity_name, intent=intent_name)``
  will look for an entity called ``entity_name`` to fill a slot
  ``slot_name`` regardless of user intent if ``intent_name`` is ``None``
  else only if the users intent is ``intent_name``.
- ``self.from_intent(intent=intent_name, value=value)``
  will fill slot ``slot_name`` with ``value`` if user intent is ``intent_name``.
  To make a boolean slot, take a look at the definition of ``outdoor_seating``
  above.
- ``self.from_text(intent=intent_name)`` will use the next
  user utterance to fill the text slot ``slot_name`` regardless of user intent
  if ``intent_name`` is ``None`` else only if user intent is ``intent_name``.
- If you want to allow a combination of these, provide them as a list as in the
  example above


Validating user input
---------------------

After extracting a slot value from user input, the form will try to validate the 
value of the slot. By default, this only checks if the requested slot was extracted.
If you want to add custom validation, for example to check a value against a database, 
you can do this by overwriting the ``validate()`` method. 
Here is an example which checks if the extracted cuisine slot belongs to a 
list of supported cuisines.

.. code-block:: python

    @staticmethod
    def cuisine_db():
        # type: () -> List[Text]
        """Database of supported cuisines"""
        return ["caribbean", "chinese", "french", "greek", "indian",
                "italian", "mexican"]

    def validate(self, dispatcher, tracker, domain):
        # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
        """"Validate extracted requested slot else raise an error"""
        slot_to_fill = tracker.get_slot(REQUESTED_SLOT)

        # extract requested slot from a user input by using `slot_mapping`
        events = self.extract(dispatcher, tracker, domain)
        if events is None:
            # raise an error if nothing was extracted
            raise ActionExecutionRejection(self.name(),
                                           "Failed to validate slot {0} "
                                           "with action {1}"
                                           "".format(slot_to_fill,
                                                     self.name()))

        extracted_slots = []
        validated_events = []
        for e in events:
            if e['event'] == 'slot':
                # get values of extracted slots to validate them later
                extracted_slots.append(e['value'])
            else:
                # add other events without validating them
                validated_events.append(e)

        for slot in extracted_slots:
            if slot_to_fill == 'cuisine':
                if slot.lower() not in self.cuisine_db():
                    dispatcher.utter_template('utter_wrong_cuisine', tracker)
                    # validation failed, set this slot to None, meaning the
                    user will be asked for the slot again
                    validated_events.append(SlotSet(slot_to_fill, None))
                else:
                    # validation succeeded
                    validated_events.append(SlotSet(slot_to_fill, slot))

            else:
                # no validation needed
                validated_events.append(SlotSet(slot_to_fill, slot))

        return validated_events

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

Handling conditional slot logic
-------------------------------

Many forms require more logic than just requesting a list of fields.
For example, if someone requests ``greek`` as their cuisine, you may want to
ask if they are looking for somewhere with outside seating.

You can achieve this by writing some logic into the ``required_slots()`` method,
for example:

.. code-block:: python

    @staticmethod
    def required_slots(tracker):
       # type: () -> List[Text]
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

The first thing to try is to run your bot with the ``debug`` flag, see :ref:`debugging` for details.
If you are just getting started, you probably only have a few hand-written stories.
This is a great starting point, but
you should give your bot to people to test **as soon as possible**. One of the guiding principles
behind Rasa Core is:

.. pull-quote:: Learning from real conversations is more important than designing hypothetical ones

So don't try to cover every possibility in your hand-written stories before giving it to testers.
Real user behavior will always surprise you!


.. include:: feedback.inc
