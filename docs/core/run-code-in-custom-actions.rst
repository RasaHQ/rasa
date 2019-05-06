:desc: Read more about how to run custom actions and code with open source Rasa
       Stack to integrate your contextual Assistant in your processes and databases.

.. _actions:

Actions
=======

.. contents::

Actions are the things your bot runs in response to user input.
There are three kinds of actions in Rasa Core:

 1. **default actions** (``action_listen``, ``action_restart``,
    ``action_default_fallback``)
 2. **utterance actions**, starting with ``utter_``, which just send a message
    to the user (see :ref:`responses`).
 3. **custom actions** - any other action, these actions can run arbitrary code

Utterance Actions
-----------------

To define an utterance action (``ActionUtterTemplate``), add an utterance template to the domain file
that starts with ``utter_``:

.. code-block:: yaml

    templates:
      utter_my_message:
        - "this is what I want my action to say!"

It is conventional to start the name of an utterance action with ``utter_``.
If this prefix is missing, you can still use the template in your custom
actions, but the template can not be directly predicted as its own action.
See :ref:`responses` for more details.

If you use an external NLG service, you don't need to specify the
templates in the domain, but you still need to add the utterance names
to the actions list of the domain.

.. _customactions:

Custom Actions
--------------

An action can run any code you want. Custom actions can turn on the lights,
add an event to a calendar, check a user's bank balance, or anything
else you can imagine.

Core will call an endpoint you can specify, when a custom action is
predicted. This endpoint should be a webserver that reacts to this
call, runs the code and optionally returns information to modify
the dialogue state.

To specify, your action server use the ``endpoints.yml``:

.. code-block:: yaml

   action_endpoint:
     url: "http://localhost:5055/webhook"

And pass it to the scripts using ``--endpoints endpoints.yml``.

You can create an action server in node.js, .NET, java, or any
other language and define your actions there - but we provide
a small python sdk to make development there even easier.

Custom Actions Written in Python
--------------------------------

For actions written in python, we have a convenient SDK which starts
this action server for you.

The only thing your action server needs to install is ``rasa_core_sdk``:

.. code-block:: bash

    pip install rasa_core_sdk

.. note::

    You do not need to install ``rasa`` for your action server.
    E.g. it is recommended to run Rasa Core in a docker container and
    create a separate container for your action server. In this
    separate container, you only need to install ``rasa_core_sdk``.

If your actions are defined in a file
called ``actions.py``, run this command:

.. code-block:: bash

    python -m rasa_core_sdk.endpoint --actions actions

.. _custom_action_example:

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
         # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict[Text, Any]]

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
``rasa_core_sdk.executor.CollectingDispatcher`` method.

Details of the ``run`` method:

.. automethod:: rasa_core_sdk.Action.run


There is an example of a ``SlotSet`` event
:ref:`below <custom_action_example>`, and a full list of possible
events in :ref:`events`.

Default Actions
---------------

There are four default actions:

+-----------------------------+------------------------------------------------+
| ``action_listen``           | Stop predicting more actions and wait for user |
|                             | input                                          |
+-----------------------------+------------------------------------------------+
| ``action_restart``          | Reset the whole conversation, usually triggered|
|                             | by using ``/restart``                          |
+-----------------------------+------------------------------------------------+
| ``action_default_fallback`` | Undoes the last user message (as if the user   |
|                             | did not send it) and utters a message that the |
|                             | bot did not understand. See :ref:`fallbacks`.  |
+-----------------------------+------------------------------------------------+
|``action_back``              | Triggered by typing ``/back``. Undoes the last |
|                             | user message (as if the user did not send it)  |
|                             | and utters a message to the user if one is     |
|                             | provided as ``utter_back`` in the domain.      |
+-----------------------------+------------------------------------------------+

All the default actions can be overwritten. To do so, add the action name
to the list of actions in your domain:

.. code-block:: yaml

  actions:
  - action_listen

Rasa Core will then call your action endpoint and treat it as every other
custom action.

Execute Actions in Other Code
-----------------------------

Rasa Core will send an HTTP ``POST`` request to your server containing
information on which action to run. Furthermore, this request will contain all
information about the conversation.

As a response to the action call from Core, you can modify the tracker,
e.g. by setting slots and send responses back to the user.
All of the modifications are done using events.
There is a list of all possible event types in :ref:`events`.


Action Server API
~~~~~~~~~~~~~~~~~

Documentation of the action server API as
:download:`OpenAPI Spec <_static/spec/action_server.yml>`.

.. apidoc::
   :path: ../_static/spec/action_server.yml

.. include:: feedback.inc
