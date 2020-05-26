:desc: Extend your Rasa conversational AI assistant using Rasa-SDK to connect to a
       external APIs or improve dialogue with custom actions written in Python. a
 a
.. _rasa-sdk: a
 a
Rasa SDK a
======== a
 a
.. edit-link:: a
 a
Rasa SDK provides the tools you need to write custom actions in python. a
 a
.. contents:: a
   :local: a
 a
Installation a
------------ a
 a
Use ``pip`` to install ``rasa-sdk`` on your action server. a
 a
.. code-block:: bash a
 a
    pip install rasa-sdk a
 a
.. note:: a
 a
    You do not need to install ``rasa`` for your action server. a
    E.g. if you are running Rasa in a docker container, it is recommended to a
    create a separate container for your action server. In this a
    separate container, you only need to install ``rasa-sdk``. a
 a
Running the Action Server a
------------------------- a
 a
If you have ``rasa`` installed, run this command to start your action server: a
 a
.. code-block:: bash a
 a
    rasa run actions a
 a
Otherwise, if you do not have ``rasa`` installed, run this command: a
 a
.. code-block:: bash a
 a
    python -m rasa_sdk --actions actions a
 a
You can verify that the action server is up and running with the command: a
 a
.. code-block:: bash a
 a
    curl http://localhost:5055/health a
 a
You can get the list of registered custom actions with the command: a
 a
.. code-block:: bash a
 a
    curl http://localhost:5055/actions a
 a
 a
The file that contains your custom actions should be called ``actions.py``. a
Alternatively, you can use a package directory called ``actions`` or else a
manually specify an actions module or package with the ``--actions`` flag. a
 a
The full list of options for running the action server with either command is: a
 a
.. program-output:: rasa run actions --help a
 a
Actions a
------- a
 a
The ``Action`` class is the base class for any custom action. It has two methods a
that both need to be overwritten, ``name()`` and ``run()``. a
 a
.. _custom_action_example: a
 a
In a restaurant bot, if the user says "show me a Mexican restaurant", a
your bot could execute the action ``ActionCheckRestaurants``, a
which might look like this: a
 a
.. testcode:: a
 a
   from rasa_sdk import Action a
   from rasa_sdk.events import SlotSet a
 a
   class ActionCheckRestaurants(Action): a
      def name(self) -> Text: a
         return "action_check_restaurants" a
 a
      def run(self, a
              dispatcher: CollectingDispatcher, a
              tracker: Tracker, a
              domain: Dict[Text, Any]) -> List[Dict[Text, Any]]: a
 a
         cuisine = tracker.get_slot('cuisine') a
         q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine) a
         result = db.query(q) a
 a
         return [SlotSet("matches", result if result is not None else [])] a
 a
 a
You should add the action name ``action_check_restaurants`` to a
the actions in your domain file. The action's ``run()`` method receives a
three arguments. You can access the values of slots and the latest message a
sent by the user using the ``tracker`` object, and you can send messages a
back to the user with the ``dispatcher`` object, by calling a
``dispatcher.utter_message``. a
 a
Details of the ``run()`` method: a
 a
.. automethod:: rasa_sdk.Action.run a
 a
Details of the ``dispatcher.utter_message()`` method: a
 a
.. automethod:: rasa_sdk.executor.CollectingDispatcher.utter_message a
 a
 a
.. _custom_session_start: a
 a
Customising the session start action a
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ a
 a
The default behaviour of the session start action is to take all existing slots and to a
carry them over into the next session. Let's say you do not want to carry over all a
slots, but only a user's name and their phone number. To do that, you'd override the a
``action_session_start`` with a custom action that might look like this: a
 a
.. testcode:: a
 a
  from typing import Text, List, Dict, Any a
 a
  from rasa_sdk import Action, Tracker a
  from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType a
  from rasa_sdk.executor import CollectingDispatcher a
 a
 a
  class ActionSessionStart(Action): a
      def name(self) -> Text: a
          return "action_session_start" a
 a
      @staticmethod a
      def fetch_slots(tracker: Tracker) -> List[EventType]: a
          """Collect slots that contain the user's name and phone number.""" a
 a
          slots = [] a
 a
          for key in ("name", "phone_number"): a
              value = tracker.get_slot(key) a
              if value is not None: a
                  slots.append(SlotSet(key=key, value=value)) a
 a
          return slots a
 a
      async def run( a
          self, a
          dispatcher: CollectingDispatcher, a
          tracker: Tracker, a
          domain: Dict[Text, Any], a
      ) -> List[EventType]: a
 a
          # the session should begin with a `session_started` event a
          events = [SessionStarted()] a
 a
          # any slots that should be carried over should come after the a
          # `session_started` event a
          events.extend(self.fetch_slots(tracker)) a
 a
          # an `action_listen` should be added at the end as a user message follows a
          events.append(ActionExecuted("action_listen")) a
 a
          return events a
 a
.. note:: a
 a
  You need to explicitly add ``action_session_start`` to your domain to override this a
  custom action. a
 a
Events a
------ a
 a
An action's ``run()`` method returns a list of events. For more information on a
the different types of events, see :ref:`Events`. There is an example of a ``SlotSet`` event a
:ref:`above <custom_action_example>`. The action itself will automatically be added to the a
tracker as an ``ActionExecuted`` event. If the action should not trigger any a
other events, it should return an empty list. a
 a
Tracker a
------- a
 a
The ``rasa_sdk.Tracker`` lets you access the bot's memory in your custom a
actions. You can get information about past events and the current state of the a
conversation through ``Tracker`` attributes and methods. a
 a
The following are available as attributes of a ``Tracker`` object: a
 a
- ``sender_id`` - The unique ID of person talking to the bot. a
- ``slots`` - The list of slots that can be filled as defined in the a
  "ref"`domains`. a
- ``latest_message`` - A dictionary containing the attributes of the latest a
  message: ``intent``, ``entities`` and ``text``. a
- ``events`` - A list of all previous events. a
- ``active_form`` - The name of the currently active form. a
- ``latest_action_name`` - The name of the last action the bot executed. a
 a
The available methods from the ``Tracker`` are: a
 a
.. automethod:: rasa_sdk.interfaces.Tracker.current_state a
 a
.. automethod:: rasa_sdk.interfaces.Tracker.is_paused a
 a
.. automethod:: rasa_sdk.interfaces.Tracker.get_latest_entity_values a
 a
.. automethod:: rasa_sdk.interfaces.Tracker.get_latest_input_channel a
 a
.. automethod:: rasa_sdk.interfaces.Tracker.events_after_latest_restart a
 a
.. automethod:: rasa_sdk.interfaces.Tracker.get_slot a
 a