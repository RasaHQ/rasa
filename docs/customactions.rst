.. _customactions:

Custom Actions
==============

There are two kinds of actions in Rasa Core. 
The simplest are ``UtterActions``, which just send a message to the user.
You define them by adding an entry to the action list in your :class:`rasa_core.domain.Domain`.
There also needs to be a matching utterance. For example, if there's an action ``utter_greet``
then there should also be an utterance template called ``utter_greet`` in your domain.

**What about more complicated actions?**
In general, an action can run any code you like. Custom actions can turn on the lights,
add an event to a calendar, check a user's bank balance, or anything else you can imagine.


Custom Actions Written in Python
--------------------------------

In a restaurant bot, if the user says "show me a Mexican restaurant",
your bot could execute the action ``ActionCheckRestaurants``, which might look like this:


.. testcode::

   from rasa_core.actions import Action
   from rasa_core.events import SlotSet

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


The action's ``run`` method receives three arguments. You can access the values of slots and
the latest message sent by the user using the ``tracker`` object, and you can send messages
back to the user with the ``dispatcher`` object, by calling ``dispatcher.utter_template``,  
``dispatcher.utter_message``, or any other :class:`Dispatcher` method. 

Details of the ``run`` method:

.. automethod:: rasa_core.actions.Action.run



