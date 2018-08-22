.. _customactions:

Custom Actions
==============


There are two kinds of actions in Rasa Core.
The simplest is an ``UtterAction``, which just sends a message to the user
(see :ref:`responses`).
To define an ``UtterAction``, add an utterance template to the domain file,
that starts with ``utter_``:

.. code-block:: yaml

    templates:
      utter_my_message:
        - "this is what I want my action to say!"

It is conventional to start the name of an ``UtterAction`` with ``utter_``.
If this prefix is missing, you can still use the template in your custom
actions, but the template can not be directly predicted as its own action.
See :ref:`responses` for more details.

Actions Which Execute Code
--------------------------

An action can run any code you want. 
Custom actions can turn on the lights,
add an event to a calendar, check a user's bank balance, or anything else you can imagine.

Rasa Core will tell your server which action to execute. 
To tell Rasa Core what happened when an action was executed, an action can return a list of ``events``.
There is an example of a ``SlotSet`` event :ref:`below <custom_action_example>` , and a full list of possible
events in :ref:`events`.


For actions written in python, we have a convenient SDK which starts this action server for you.
If your actions are defined in a file called ``actions.py``, run this command:

.. code-block:: bash

    python -m rasa_core_sdk.endpoint --actions actions

However, you can also create a server in node.js, .NET, java, or any other language and define your acitons there.

Whichever option you go for, you will then need to add an entry into your
``endpoints.yml`` as follows:

.. code-block:: yaml

   action_endpoint:
     url: "http://localhost:5055/webhook"

.. _custom_action_example:

Custom Actions Written in Python
--------------------------------

In a restaurant bot, if the user says "show me a Mexican restaurant",
your bot could execute the action ``ActionCheckRestaurants``,
which might look like this:



.. testcode::

   from rasa_core_sdk import Action
   from rasa_core_sdk.events import SlotSet

   class ActionCheckRestaurants(Action):
      def name(self):
         # type: () -> Text
         return "action_check_restaurants"

      def run(self, dispatcher, tracker, domain):
         # type: (Dispatcher, DialogueStateTracker, Domain) -> List[Event]

         cuisine = tracker.get_slot('cuisine')
         q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)
         result = db.query(q)

         return [SlotSet("matches", result if result is not None else [])]


You should add the the action name ``action_check_restaurants`` to the actions in your domain file.
The action's ``run`` method receives three arguments. You can access the values of slots and
the latest message sent by the user using the ``tracker`` object, and you can send messages
back to the user with the ``dispatcher`` object, by calling ``dispatcher.utter_template``,
``dispatcher.utter_message``, or any other :class:`Dispatcher` method.

Details of the ``run`` method:

.. automethod:: rasa_core.actions.Action.run
