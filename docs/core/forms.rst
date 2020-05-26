:desc: Follow a rule-based process of information gathering using FormActions a
       in open source bot framework Rasa. a
 a
.. _forms: a
 a
Forms a
===== a
 a
.. edit-link:: a
 a
.. note:: a
   There is an in-depth tutorial `here <https://blog.rasa.com/building-contextual-assistants-with-rasa-formaction/>`_ about how to use Rasa Forms for slot filling. a
 a
.. contents:: a
   :local: a
 a
One of the most common conversation patterns is to collect a few pieces of a
information from a user in order to do something (book a restaurant, call an a
API, search a database, etc.). This is also called **slot filling**. a
 a
 a
If you need to collect multiple pieces of information in a row, we recommended a
that you create a ``FormAction``. This is a single action which contains the a
logic to loop over the required slots and ask the user for this information. a
There is a full example using forms in the ``examples/formbot`` directory of a
Rasa Core. a
 a
 a
When you define a form, you need to add it to your domain file. a
If your form's name is ``restaurant_form``, your domain would look like this: a
 a
.. code-block:: yaml a
 a
   forms: a
     - restaurant_form a
   actions: a
     ... a
 a
See ``examples/formbot/domain.yml`` for an example. a
 a
Configuration File a
------------------ a
 a
To use forms, you also need to include the ``FormPolicy`` in your policy a
configuration file. For example: a
 a
.. code-block:: yaml a
 a
  policies: a
    - name: "FormPolicy" a
 a
see ``examples/formbot/config.yml`` for an example. a
 a
Form Basics a
----------- a
 a
Using a ``FormAction``, you can describe *all* of the happy paths with a single a
story. By "happy path", we mean that whenever you ask a user for some information, a
they respond with the information you asked for. a
 a
If we take the example of the restaurant bot, this single story describes all of the a
happy paths. a
 a
.. code-block:: story a
 a
    ## happy path a
    * request_restaurant a
        - restaurant_form a
        - form{"name": "restaurant_form"} a
        - form{"name": null} a
 a
In this story the user intent is ``request_restaurant``, which is followed by a
the form action ``restaurant_form``. With ``form{"name": "restaurant_form"}`` the a
form is activated and with ``form{"name": null}`` the form is deactivated again. a
As shown in the section :ref:`section_unhappy` the bot can execute any kind of a
actions outside the form while the form is still active. On the "happy path", a
where the user is cooperating well and the system understands the user input correctly, a
the form is filling all requested slots without interruption. a
 a
The ``FormAction`` will only request slots which haven't already been set. a
If a user starts the conversation with a
`I'd like a vegetarian Chinese restaurant for 8 people`, then they won't be a
asked about the ``cuisine`` and ``num_people`` slots. a
 a
Note that for this story to work, your slots should be :ref:`unfeaturized a
<unfeaturized-slot>`. If any of these slots are featurized, your story needs to a
include ``slot{}`` events to show these slots being set. In that case, the a
easiest way to create valid stories is to use :ref:`interactive-learning`. a
 a
In the story above, ``restaurant_form`` is the name of our form action. a
Here is an example of what it looks like. a
You need to define three methods: a
 a
- ``name``: the name of this action a
- ``required_slots``: a list of slots that need to be filled for the ``submit`` method to work. a
- ``submit``: what to do at the end of the form, when all the slots have been filled. a
 a
.. literalinclude:: ../../examples/formbot/actions.py a
   :dedent: 4 a
   :pyobject: RestaurantForm.name a
 a
.. literalinclude:: ../../examples/formbot/actions.py a
   :dedent: 4 a
   :pyobject: RestaurantForm.required_slots a
 a
.. literalinclude:: ../../examples/formbot/actions.py a
   :dedent: 4 a
   :pyobject: RestaurantForm.submit a
 a
Once the form action gets called for the first time, a
the form gets activated and the ``FormPolicy`` jumps in. a
The ``FormPolicy`` is extremely simple and just always predicts the form action. a
See :ref:`section_unhappy` for how to work with unexpected user input. a
 a
Every time the form action gets called, it will ask the user for the next slot in a
``required_slots`` which is not already set. a
It does this by looking for a response called ``utter_ask_{slot_name}``, a
so you need to define these in your domain file for each required slot. a
 a
Once all the slots are filled, the ``submit()`` method is called, where you can a
use the information you've collected to do something for the user, for example a
querying a restaurant API. a
If you don't want your form to do anything at the end, just use ``return []`` a
as your submit method. a
After the submit method is called, the form is deactivated, a
and other policies in your Core model will be used to predict the next action. a
 a
Custom slot mappings a
-------------------- a
 a
If you do not define slot mappings, slots will be only filled by entities a
with the same name as the slot that are picked up from the user input. a
Some slots, like ``cuisine``, can be picked up using a single entity, but a a
``FormAction`` can also support yes/no questions and free-text input. a
The ``slot_mappings`` method defines how to extract slot values from user responses. a
 a
Here's an example for the restaurant bot: a
 a
.. literalinclude:: ../../examples/formbot/actions.py a
   :dedent: 4 a
   :pyobject: RestaurantForm.slot_mappings a
 a
The predefined functions work as follows: a
 a
- ``self.from_entity(entity=entity_name, intent=intent_name, role=role_name, group=group_name)`` a
  will look for an entity called ``entity_name`` to fill a slot a
  ``slot_name`` regardless of user intent if ``intent_name`` is ``None`` a
  else only if the users intent is ``intent_name``. If ``role_name`` and/or ``group_name`` a
  are provided, the role/group label of the entity also needs to match the given values. a
- ``self.from_intent(intent=intent_name, value=value)`` a
  will fill slot ``slot_name`` with ``value`` if user intent is ``intent_name``. a
  To make a boolean slot, take a look at the definition of ``outdoor_seating`` a
  above. Note: Slot will not be filled with user intent of message triggering a
  the form action. Use ``self.from_trigger_intent`` below. a
- ``self.from_trigger_intent(intent=intent_name, value=value)`` a
  will fill slot ``slot_name`` with ``value`` if form was triggered with user a
  intent ``intent_name``. a
- ``self.from_text(intent=intent_name)`` will use the next a
  user utterance to fill the text slot ``slot_name`` regardless of user intent a
  if ``intent_name`` is ``None`` else only if user intent is ``intent_name``. a
- If you want to allow a combination of these, provide them as a list as in the a
  example above a
 a
 a
Validating user input a
--------------------- a
 a
After extracting a slot value from user input, the form will try to validate the a
value of the slot. Note that by default, validation only happens if the form a
action is executed immediately after user input. This can be changed in the a
``_validate_if_required()`` function of the ``FormAction`` class in Rasa SDK. a
Any required slots that were filled before the initial activation of a form a
are validated upon activation as well. a
 a
By default, validation only checks if the requested slot was successfully a
extracted from the slot mappings. If you want to add custom validation, for a
example to check a value against a database, you can do this by writing a helper a
validation function with the name ``validate_{slot-name}``. a
 a
Here is an example , ``validate_cuisine()``, which checks if the extracted cuisine slot a
belongs to a list of supported cuisines. a
 a
.. literalinclude:: ../../examples/formbot/actions.py a
   :pyobject: RestaurantForm.cuisine_db a
 a
.. literalinclude:: ../../examples/formbot/actions.py a
   :pyobject: RestaurantForm.validate_cuisine a
 a
As the helper validation functions return dictionaries of slot names and values a
to set, you can set more slots than just the one you are validating from inside a
a helper validation method. However, you are responsible for making sure that a
those extra slot values are valid. a
 a
In case the slot is filled with something that you are certain can't be handled a
and you want to deactivate the form directly, a
you can overwrite the ``request_next_slot()`` method to do so. The example below a
checks the value of the ``cuisine`` slot directly, but you could use any logic a
you'd like to trigger deactivation: a
 a
.. code-block:: python a
 a
    def request_next_slot( a
        self, a
        dispatcher: "CollectingDispatcher", a
        tracker: "Tracker", a
        domain: Dict[Text, Any], a
    ) -> Optional[List[EventType]]: a
        """Request the next slot and utter template if needed, a
            else return None""" a
        for slot in self.required_slots(tracker): a
            if self._should_request_slot(tracker, slot): a
 a
                ## Condition of validated slot that triggers deactivation a
                if slot == "cuisine" and tracker.get_slot("cuisine") == "caribbean": a
                    dispatcher.utter_message(text="Sorry, I can't help you with that") a
                    return self.deactivate() a
                 a
                ## For all other slots, continue as usual a
                logger.debug(f"Request next slot '{slot}'") a
                dispatcher.utter_message( a
                    template=f"utter_ask_{slot}", **tracker.slots a
                ) a
                return [SlotSet(REQUESTED_SLOT, slot)] a
        return None a
 a
 a
If nothing is extracted from the user's utterance for any of the required slots, an a
``ActionExecutionRejection`` error will be raised, meaning the action execution a
was rejected and therefore Core will fall back onto a different policy to a
predict another action. a
 a
.. _section_unhappy: a
 a
Handling unhappy paths a
---------------------- a
 a
Of course your users will not always respond with the information you ask of them. a
Typically, users will ask questions, make chitchat, change their mind, or otherwise a
stray from the happy path. The way this works with forms is that a form will raise a
an ``ActionExecutionRejection`` if the user didn't provide the requested information. a
You need to handle events that might cause ``ActionExecutionRejection`` errors a
in your stories. For example, if you expect your users to chitchat with your bot, a
you could add a story like this: a
 a
.. code-block:: story a
 a
    ## chitchat a
    * request_restaurant a
        - restaurant_form a
        - form{"name": "restaurant_form"} a
    * chitchat a
        - utter_chitchat a
        - restaurant_form a
        - form{"name": null} a
 a
In some situations, users may change their mind in the middle of form action a
and decide not to go forward with their initial request. In cases like this, the a
assistant should stop asking for the requested slots. You can handle such situations a
gracefully using a default action ``action_deactivate_form`` which will deactivate a
the form and reset the requested slot. An example story of such conversation could a
look as follows: a
 a
.. code-block:: story a
 a
    ## chitchat a
    * request_restaurant a
        - restaurant_form a
        - form{"name": "restaurant_form"} a
    * stop a
        - utter_ask_continue a
    * deny a
        - action_deactivate_form a
        - form{"name": null} a
 a
 a
It is **strongly** recommended that you build these stories using interactive learning. a
If you write these stories by hand you will likely miss important things. a
Please read :ref:`section_interactive_learning_forms` a
on how to use interactive learning with forms. a
 a
The requested_slot slot a
----------------------- a
 a
The slot ``requested_slot`` is automatically added to the domain as an a
unfeaturized slot. If you want to make it featurized, you need to add it a
to your domain file as a categorical slot. You might want to do this if you a
want to handle your unhappy paths differently depending on what slot is a
currently being asked from the user. For example, say your users respond a
to one of the bot's questions with another question, like *why do you need to know that?* a
The response to this ``explain`` intent depends on where we are in the story. a
In the restaurant case, your stories would look something like this: a
 a
.. code-block:: story a
 a
    ## explain cuisine slot a
    * request_restaurant a
        - restaurant_form a
        - form{"name": "restaurant_form"} a
        - slot{"requested_slot": "cuisine"} a
    * explain a
        - utter_explain_cuisine a
        - restaurant_form a
        - slot{"cuisine": "greek"} a
        ( ... all other slots the form set ... ) a
        - form{"name": null} a
 a
    ## explain num_people slot a
    * request_restaurant a
        - restaurant_form a
        - form{"name": "restaurant_form"} a
        - slot{"requested_slot": "num_people"} a
    * explain a
        - utter_explain_num_people a
        - restaurant_form a
        - slot{"cuisine": "greek"} a
        ( ... all other slots the form set ... ) a
        - form{"name": null} a
 a
Again, it is **strongly** recommended that you use interactive a
learning to build these stories. a
Please read :ref:`section_interactive_learning_forms` a
on how to use interactive learning with forms. a
 a
.. _conditional-logic: a
 a
Handling conditional slot logic a
------------------------------- a
 a
Many forms require more logic than just requesting a list of fields. a
For example, if someone requests ``greek`` as their cuisine, you may want to a
ask if they are looking for somewhere with outside seating. a
 a
You can achieve this by writing some logic into the ``required_slots()`` method, a
for example: a
 a
.. code-block:: python a
 a
    @staticmethod a
    def required_slots(tracker) -> List[Text]: a
       """A list of required slots that the form has to fill""" a
 a
       if tracker.get_slot('cuisine') == 'greek': a
         return ["cuisine", "num_people", "outdoor_seating", a
                 "preferences", "feedback"] a
       else: a
         return ["cuisine", "num_people", a
                 "preferences", "feedback"] a
 a
This mechanism is quite general and you can use it to build many different a
kinds of logic into your forms. a
 a
 a
 a
Debugging a
--------- a
 a
The first thing to try is to run your bot with the ``debug`` flag, see :ref:`command-line-interface` for details. a
If you are just getting started, you probably only have a few hand-written stories. a
This is a great starting point, but a
you should give your bot to people to test **as soon as possible**. One of the guiding principles a
behind Rasa Core is: a
 a
.. pull-quote:: Learning from real conversations is more important than designing hypothetical ones a
 a
So don't try to cover every possibility in your hand-written stories before giving it to testers. a
Real user behavior will always surprise you! a
 a