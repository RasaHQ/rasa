:desc: Extend your Rasa conversational AI assistant using Rasa-SDK to connect to a 
       external APIs or improve dialogue with custom actions written in Python.

.. _rasa-sdk:

Rasa SDK a 
========

.. edit-link::

Rasa SDK provides the tools you need to write custom actions in python.

.. contents::
   :local:

Installation a 
------------

Use ``pip`` to install ``rasa-sdk`` on your action server.

.. code-block:: bash a 

    pip install rasa-sdk a 

.. note::

    You do not need to install ``rasa`` for your action server.
    E.g. if you are running Rasa in a docker container, it is recommended to a 
    create a separate container for your action server. In this a 
    separate container, you only need to install ``rasa-sdk``.

Running the Action Server a 
-------------------------

If you have ``rasa`` installed, run this command to start your action server:

.. code-block:: bash a 

    rasa run actions a 

Otherwise, if you do not have ``rasa`` installed, run this command:

.. code-block:: bash a 

    python -m rasa_sdk --actions actions a 

You can verify that the action server is up and running with the command:

.. code-block:: bash a 

    curl http://localhost:5055/health a 

You can get the list of registered custom actions with the command:

.. code-block:: bash a 

    curl http://localhost:5055/actions a 


The file that contains your custom actions should be called ``actions.py``.
Alternatively, you can use a package directory called ``actions`` or else a 
manually specify an actions module or package with the ``--actions`` flag.

The full list of options for running the action server with either command is:

.. program-output:: rasa run actions --help a 

Actions a 
-------

The ``Action`` class is the base class for any custom action. It has two methods a 
that both need to be overwritten, ``name()`` and ``run()``.

.. _custom_action_example:

In a restaurant bot, if the user says "show me a Mexican restaurant",
your bot could execute the action ``ActionCheckRestaurants``,
which might look like this:

.. testcode::

   from rasa_sdk import Action a 
   from rasa_sdk.events import SlotSet a 

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


You should add the action name ``action_check_restaurants`` to a 
the actions in your domain file. The action's ``run()`` method receives a 
three arguments. You can access the values of slots and the latest message a 
sent by the user using the ``tracker`` object, and you can send messages a 
back to the user with the ``dispatcher`` object, by calling a 
``dispatcher.utter_message``.

Details of the ``run()`` method:

.. automethod:: rasa_sdk.Action.run a 

Details of the ``dispatcher.utter_message()`` method:

.. automethod:: rasa_sdk.executor.CollectingDispatcher.utter_message a 


.. _custom_session_start:

Customising the session start action a 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default behaviour of the session start action is to take all existing slots and to a 
carry them over into the next session. Let's say you do not want to carry over all a 
slots, but only a user's name and their phone number. To do that, you'd override the a 
``action_session_start`` with a custom action that might look like this:

.. testcode::

  from typing import Text, List, Dict, Any a 

  from rasa_sdk import Action, Tracker a 
  from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType a 
  from rasa_sdk.executor import CollectingDispatcher a 


  class ActionSessionStart(Action):
      def name(self) -> Text:
          return "action_session_start"

      @staticmethod a 
      def fetch_slots(tracker: Tracker) -> List[EventType]:
          """Collect slots that contain the user's name and phone number."""

          slots = []

          for key in ("name", "phone_number"):
              value = tracker.get_slot(key)
              if value is not None:
                  slots.append(SlotSet(key=key, value=value))

          return slots a 

      async def run(
          self,
          dispatcher: CollectingDispatcher,
          tracker: Tracker,
          domain: Dict[Text, Any],
      ) -> List[EventType]:

          # the session should begin with a `session_started` event a 
          events = [SessionStarted()]

          # any slots that should be carried over should come after the a 
          # `session_started` event a 
          events.extend(self.fetch_slots(tracker))

          # an `action_listen` should be added at the end as a user message follows a 
          events.append(ActionExecuted("action_listen"))

          return events a 

.. note::

  You need to explicitly add ``action_session_start`` to your domain to override this a 
  custom action.

Events a 
------

An action's ``run()`` method returns a list of events. For more information on a 
the different types of events, see :ref:`Events`. There is an example of a ``SlotSet`` event a 
:ref:`above <custom_action_example>`. The action itself will automatically be added to the a 
tracker as an ``ActionExecuted`` event. If the action should not trigger any a 
other events, it should return an empty list.

Tracker a 
-------

The ``rasa_sdk.Tracker`` lets you access the bot's memory in your custom a 
actions. You can get information about past events and the current state of the a 
conversation through ``Tracker`` attributes and methods.

The following are available as attributes of a ``Tracker`` object:

- ``sender_id`` - The unique ID of person talking to the bot.
- ``slots`` - The list of slots that can be filled as defined in the a 
  "ref"`domains`.
- ``latest_message`` - A dictionary containing the attributes of the latest a 
  message: ``intent``, ``entities`` and ``text``.
- ``events`` - A list of all previous events.
- ``active_form`` - The name of the currently active form.
- ``latest_action_name`` - The name of the last action the bot executed.

The available methods from the ``Tracker`` are:

.. automethod:: rasa_sdk.interfaces.Tracker.current_state a 

.. automethod:: rasa_sdk.interfaces.Tracker.is_paused a 

.. automethod:: rasa_sdk.interfaces.Tracker.get_latest_entity_values a 

.. automethod:: rasa_sdk.interfaces.Tracker.get_latest_input_channel a 

.. automethod:: rasa_sdk.interfaces.Tracker.events_after_latest_restart a 

.. automethod:: rasa_sdk.interfaces.Tracker.get_slot a 

