.. _slotfilling:

Slot Filling
============

.. contents::

One of the most common conversation patterns is to collect a few pieces of
information from a user in order to do something (book a restaurant, call an
API, search a database, etc.). This is also called **slot filling**.

Slot Filling with a ``FormAction``
----------------------------------

If you need to collect multiple pieces of information in a row, it is recommended
to create a ``FormAction``. You can take a look at the base class of
the FormAction below:

.. autoclass:: rasa_core_sdk.forms.FormAction

Basics
~~~~~~

This lets you have a single action that is called multiple times, rather than
separate actions for each question. The idea behind this is that you can get the
happy path of your bot up and running quickly. If we take the example of the
restaurant bot, this means your initial story will look as simple as this
(provided your slots are `unfeaturized
<https://rasa.com/docs/core/api/slots_api/#unfeaturized-slot>`_):

.. code-block:: story

    ## happy path
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - form{"name": null}

This means that even if you provide multiple slots at one, e.g. if someone says
"I'd like a vegetarian Chinese restaurant for 8 people" the slots ``cuisine``
and ``num_people`` won't get asked for.

The ``restaurant_form`` is the form action, for which you need to define only a
few methods in order to get it working: ``name()``, ``required_slots(tracker)``
and ``submit(self, dispatcher, tracker, domain)``


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

The way this works is that once the form action gets called for the first time,
the form gets activated and the ``FormPolicy`` jumps in. The ``FormPolicy`` always
predicts the active form if there was just some user input, and always predicts
waiting for user input when the active form was just called.

At each turn that the form action gets called, it will take the next slot from
the ``required_slots(tracker)`` that is still empty and set a slot called
``requested_slot`` to keep track of what it has asked the user.
This is done by using the ``utter_ask_{slot_name}`` templates that you need to
define in your domain file. By default it will fill the slot with the entity of
the same name if it's present.

Once all the slots are filled, the ``submit()`` method is called, in which you
can perform some final actions, e.g. querying a restaurant API with the
extracted slots (or you can also just do nothing by only writing ``return []``).
After this the form is deactivated, and prediction goes back to other policies
present in your Core model.

Custom slot mappings
~~~~~~~~~~~~~~~~~~~~

You can also fill slots with the full user message, map an intent to a value,
or map a slot to an entity of a different name by defining the ``slot_mapping()``
function. Here's an example for the restaurant bot:

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
~~~~~~~~~~~~~~~~~~~~~

By default validation will fail only if the slot isn't filled. If
you want to add custom validation, e.g. only accept a specific type of cuisine,
overwrite the ``validate()`` function:

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

If nothing is extracted from the users utterance for the slot, an
``ActionExecutionRejection`` error will be raised, meaning the action execution
was rejected and therefore Core will fall back onto a different policy to
predict another action.

Handling unhappy paths
~~~~~~~~~~~~~~~~~~~~~~

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

The requested_slot slot
~~~~~~~~~~~~~~~~~~~~~~~
The slot ``requested_slot`` is automatically added to the domain as an
unfeaturized slot. If you want to make it featurized, you need to add it
to your domain file. You might want to do this if you want to handle your
unhappy paths differently depending on what slot is currently being asked from
the user. For example, say you want to let your users ask for explanations of
the different slot values. Then your stories would look something like this:

.. code-block:: story

    ## explain cuisine slot
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - slot{"requested_slot": "cuisine"}
    * explain
        - utter_explain_cuisine
        - restaurant_form
        - form{"name": null}

    ## explain num_people slot
    * request_restaurant
        - restaurant_form
        - form{"name": "restaurant_form"}
        - slot{"requested_slot": "num_people"}
    * explain
        - utter_explain_num_people
        - restaurant_form
        - form{"name": null}


Handling conditional slot logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to change the lists of slots needed from the user dependent on
some Rasa Core event (e.g. the first slot has a specific value) you should
do this in the ``required_slots(tracker)`` method by returning a different list
dependent on that event. For example, say that only greek restaurants provide
outdoor seating and so you don't want to ask that for other cuisines. Then
your method would look something like this:

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


Interactive learning
~~~~~~~~~~~~~~~~~~~~

You may want to teach the bot how to handle unexpected user behaviour like
chitchat through interactive learning. Please read `these instructions
<https://rasa.com/docs/core/interactive_learning/>`_ on how to use interative
learning with forms


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


.. include:: feedback.inc
