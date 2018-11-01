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

.. code-block:: story

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

.. code-block:: yaml

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


.. code-block:: story
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
:ref:`interactive_learning` is a great way to explore more conversations
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

So don't try to cover every possibility in your hand-written stories before giving it to testers.
Real user behavior will always surprise you!


Slot Filling with a ``FormAction``
----------------------------------

If you need to collect multiple pieces of information in a row, it is sometimes easier
to create a ``FormAction``.
This lets you have a single action that is called multiple times, rather than separate actions for each question,
e.g. ``utter_ask_cuisine``, ``utter_ask_num_people``, in a restaurant bot.


.. note::
    You don't *have* to use a ``FormAction`` to do slot filling! It just means you need
    fewer stories to get the initial flow working.

A form action has a set of required slots, which you define for the class:

.. code-block:: python

 class RestaurantForm(FormAction):
    """Example of a custom form action"""

    def name(self):
        # type: () -> Text
        """Unique identifier of the form"""

        return "restaurant_form"

    @staticmethod
    def required_slots():
        # type: () -> List[Text]
        """A list of required slots that the form has to fill"""

        return ["cuisine", "num_people", "outdoor_seating",
                "preferences", "feedback"]


The way this works is that every time you call this action, it will pick one slot from the
``required_slots()`` that's still missing and ask the user for it. You can also ask a yes/no
question by providing intent - value pairs in ``slot_mapping``:

.. code-block:: python

 class RestaurantForm(FormAction):
    """Example of a custom form action"""

    ...

    def slot_mapping(self):
        # type: () -> Dict[Text: Union[Text, Dict, List[Text, Dict]]]
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

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


The form action will set a slot called ``requested_slot`` to keep track of what it has asked the user and to be able to validate arbitrary user input:

.. code-block:: python

 class RestaurantForm(FormAction):
    """Example of a custom form action"""

    ...

        @staticmethod
    def cuisine_db():
        # type: () -> List[Text]
        """Database of supported cuisines"""
        return ["caribbean",
                "chinese",
                "french",
                "greek",
                "indian",
                "italian",
                "mexican"]

    @staticmethod
    def is_int(string):
        # type: (Text) -> bool
        """Check if a string is an integer"""
        try:
            int(string)
            return True
        except ValueError:
            return False

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
                    # validation failed, set this slot to None
                    validated_events.append(SlotSet(slot_to_fill, None))
                else:
                    # validation succeeded
                    validated_events.append(SlotSet(slot_to_fill, slot))

            elif slot_to_fill == 'num_people':
                if not self.is_int(slot) or int(slot) <= 0:
                    dispatcher.utter_template('utter_wrong_num_people',
                                              tracker)
                    # validation failed, set this slot to None
                    validated_events.append(SlotSet(slot_to_fill, None))
                else:
                    # validation succeeded
                    validated_events.append(SlotSet(slot_to_fill, slot))

            elif slot_to_fill == 'outdoor_seating':
                if isinstance(slot, bool):
                    # slot already boolean
                    validated_events.append(SlotSet(slot_to_fill, slot))
                elif 'out' in slot:
                    # convert out... to True
                    validated_events.append(SlotSet(slot_to_fill, True))
                elif 'in' in slot:
                    # convert in... to False
                    validated_events.append(SlotSet(slot_to_fill, False))
                else:
                    # set a slot to whatever it is
                    validated_events.append(SlotSet(slot_to_fill, slot))

            else:
                # no validation needed
                validated_events.append(SlotSet(slot_to_fill, slot))

        return validated_events


The ``submit()`` method is called when the action is run and all slots are filled.

.. code-block:: python

 class RestaurantForm(FormAction):
    """Example of a custom form action"""

    ...

    def submit(self, dispatcher, tracker, domain):
    # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
    """Define what the form has to do
        after all required slots are filled"""

    # utter submit template
    dispatcher.utter_template('utter_submit', tracker)
    return []



Important is that a story should only contain the first call to ``FormAction``,
so a story will look something like this, if all slots are ``unfeaturized``:

.. code-block:: story

    ## happy path
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - form{"name": null}


Some important things to consider:

- If you are just collecting some information and don't need to make an API call at the end, your ``submit()`` method
  should just ``return []``.
- The slot ``requested_slot`` is automatically added to the domain as an unfeaturized slot. If you want to make it featurized, you need to add it to your domain file.
- You need to define utterances for asking for each slot in your domain, e.g. ``utter_ask_{slot_name}``.
- We strongly recommend that you create these stories using interactive learning, because if you
  type these by hand you will probably forget to include the lines for the ``requested_slot``.
- Any slots that are already set won't be asked for. E.g. if someone says "I'd like a vegetarian Chinese restaurant for 8 people" the ``submit`` function should get called right away.


Setting categorical slots and Free-text Input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pre-defined ``slot_mapping`` types are:

- ``{slot_name: self.from_entity(entity=entity_name, intent=intent_name)}``, which will
  look for an entity called ``entity_name`` to fill a slot ``slot_name`` regardless of user intent if ``intent_name`` is ``None`` else if user intent is ``intent_name``.
- ``{slot_name: self.from_intent(intent=intent_name, value=value)}``, which will fill slot ``slot_name`` with ``value`` if user intent is ``intent_name``.
  To make a slot boolean use ``{boolean_slot_name: [self.from_intent(intent=affirm_intent, value=True), self.from_intent(intent=deny_intent, value=False)]}``
  which looks for the intent ``affirm_intent`` and ``deny_intent`` to fill a boolean slot called ``boolean_slot_name``.
- ``{slot_name: self.from_text(intent=intent_name)}``, which will use the next user utterance to fill the text slot ``slot_name`` regardless of user intent if ``intent_name`` is ``None`` else if user intent is ``intent_name``.


Before setting a slot``FormAction`` will call ``self.validate(...)``. By default this just checks that the slot can be extracted, but if you want to check
the value against a DB, or check a pattern is matched, you can do so by overriding the ``validate(...)`` method.

.. include:: feedback.inc
