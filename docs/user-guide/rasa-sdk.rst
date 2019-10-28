:desc: Information about `rasa-sdk` objects and functions.

.. _rasa-sdk:

Rasa-SDK
=======

.. edit-link::

``Rasa-SDK`` provides the tools you need to write :ref:`custom actions`.

.. contents::
   :local:

``Action``
----------

The ``Action`` class is the base class for any custom action. It has two methods
that both need to be overwritten, `.name()` and `.run()`.

.. _custom_action_example_verbose:

In a restaurant bot, if the user says "show me a Mexican restaurant",
your bot could execute the action ``ActionCheckRestaurants``,
which might look like this:

.. testcode::

   from rasa_sdk import Action
   from rasa_sdk.events import SlotSet

   class ActionCheckRestaurants(Action):
      def name(self) -> Text:
         return "action_check_restaurants"

      def run(self,
              dispatcher: CollectingDispatcher,
              tracker: Tracker,
              domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

         cuisine = tracker.get_slot('cuisine')
         q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)
         result = db.query(q)

         return [SlotSet("matches", result if result is not None else [])]


You should add the the action name ``action_check_restaurants`` to
the actions in your domain file. The action's ``run`` method receives
three arguments. You can access the values of slots and the latest message
sent by the user using the ``tracker`` object, and you can send messages
back to the user with the ``dispatcher`` object, by calling
``dispatcher.utter_template``, ``dispatcher.utter_message``, or any other
``rasa_sdk.executor.CollectingDispatcher`` method.

Details of the ``run()`` method:

.. automethod:: rasa_sdk.Action.run


There is an example of a ``SlotSet`` event
:ref:`above <custom_action_example>`, and a full list of possible
events in :ref:`Events <events>`.

``Tracker``
-----------

The ``rasa_sdk.Tracker`` lets you access the bots memory in your custom
actions. You can get information about past events and the current state of the
conversation through ``Tracker`` attributes.

In no particular order the attributes of the ``Tracker`` are:
- ``sender_id`` - The unique ID of person talking to the bot.
- ``slots`` - The list of slots that can be filled as defined in the
  "ref"`domains`.
- ``latest_message`` - A dictionary containing the attributes of the latest
  message: ``intent``, ``entities`` and ``text``.
- ``latest_event_time`` - The timestap from the last event that occured.
- ``paused`` - Whether the tracker is currently paused.
- ``events`` - A list of all previous events.
- ``latest_input_channel`` - The name of the input channel of the last
  ``UserUttered`` event.
- ``active_form`` - The name of the currently active form.
- ``latest_action_name`` - The name of the last action the bot executed.

