:desc: Extend your Rasa conversational AI assistant using Rasa-SDK to connect to
       external APIs or improve dialogue with custom actions written in Python.

.. _rasa-sdk:

Rasa SDK
========

.. edit-link::

Rasa SDK provides the tools you need to write custom actions in python.

.. contents::
   :local:

Installation
------------

Use ``pip`` to install ``rasa-sdk`` on your action server.

.. code-block:: bash

    pip install rasa-sdk

.. note::

    You do not need to install ``rasa`` for your action server.
    E.g. if you are running Rasa in a docker container, it is recommended to
    create a separate container for your action server. In this
    separate container, you only need to install ``rasa-sdk``.

Running the Action Server
-------------------------

If you have ``rasa`` installed, run this command to start your action server:

.. code-block:: bash

    rasa run actions

Otherwise, if you do not have ``rasa`` installed, run this command:

.. code-block:: bash

    python -m rasa_sdk --actions actions

You can verify that the action server is up and running with the command:

.. code-block:: bash

    curl http://localhost:5055/health

You can get the list of registered custom actions with the command:

.. code-block:: bash

    curl http://localhost:5055/actions


The file that contains your custom actions should be called ``actions.py``.
Alternatively, you can use a package directory called ``actions`` or else
manually specify an actions module or package with the ``--actions`` flag.

The full list of options for running the action server with either command is:

.. program-output:: rasa run actions --help

Actions
-------

The ``Action`` class is the base class for any custom action. It has two methods
that both need to be overwritten, ``name()`` and ``run()``.

.. _custom_action_example:

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
the actions in your domain file. The action's ``run()`` method receives
three arguments. You can access the values of slots and the latest message
sent by the user using the ``tracker`` object, and you can send messages
back to the user with the ``dispatcher`` object, by calling
``dispatcher.utter_message``.

Details of the ``run()`` method:

.. automethod:: rasa_sdk.Action.run

Details of the ``dispatcher.utter_message()`` method:

.. automethod:: rasa_sdk.executor.CollectingDispatcher.utter_message


.. _custom_session_start:

Customising the session start action
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default behaviour of the session start action is to take all existing slots and to
carry them over into the next session. Let's say you do not want to carry over all
slots, but only a user's name and their phone number. To do that, you'd override the
``action_session_start`` with a custom action that might look like this:

.. testcode::

  from typing import Text, List, Dict, Any

  from rasa_sdk import Action, Tracker
  from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType
  from rasa_sdk.executor import CollectingDispatcher


  class ActionSessionStart(Action):
      def name(self) -> Text:
          return "action_session_start"

      @staticmethod
      def fetch_slots(tracker: Tracker) -> List[EventType]:
          """Collect slots that contain the user's name and phone number."""

          slots = []

          for key in ("name", "phone_number"):
              value = tracker.get_slot(key)
              if value is not None:
                  slots.append(SlotSet(key=key, value=value))

          return slots

      async def run(
          self,
          dispatcher: CollectingDispatcher,
          tracker: Tracker,
          domain: Dict[Text, Any],
      ) -> List[EventType]:

          # the session should begin with a `session_started` event
          events = [SessionStarted()]

          # any slots that should be carried over should come after the
          # `session_started` event
          events.extend(self.fetch_slots(tracker))

          # an `action_listen` should be added at the end as a user message follows
          events.append(ActionExecuted("action_listen"))

          return events

.. note::

  You need to explicitly add ``action_session_start`` to your domain to override this
  custom action.

Events
------

An action's ``run()`` method returns a list of events. For more information on
the different types of events, see :ref:`Events`. There is an example of a ``SlotSet`` event
:ref:`above <custom_action_example>`. The action itself will automatically be added to the
tracker as an ``ActionExecuted`` event. If the action should not trigger any
other events, it should return an empty list.

Tracker
-------

The ``rasa_sdk.Tracker`` lets you access the bot's memory in your custom
actions. You can get information about past events and the current state of the
conversation through ``Tracker`` attributes and methods.

The following are available as attributes of a ``Tracker`` object:

- ``sender_id`` - The unique ID of person talking to the bot.
- ``slots`` - The list of slots that can be filled as defined in the
  "ref"`domains`.
- ``latest_message`` - A dictionary containing the attributes of the latest
  message: ``intent``, ``entities`` and ``text``.
- ``events`` - A list of all previous events.
- ``active_form`` - The name of the currently active form.
- ``latest_action_name`` - The name of the last action the bot executed.

The available methods from the ``Tracker`` are:

.. automethod:: rasa_sdk.interfaces.Tracker.current_state

.. automethod:: rasa_sdk.interfaces.Tracker.is_paused

.. automethod:: rasa_sdk.interfaces.Tracker.get_latest_entity_values

.. automethod:: rasa_sdk.interfaces.Tracker.get_latest_input_channel

.. automethod:: rasa_sdk.interfaces.Tracker.events_after_latest_restart

.. automethod:: rasa_sdk.interfaces.Tracker.get_slot
