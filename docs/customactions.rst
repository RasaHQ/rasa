.. _customactions:

Custom Actions
==============


The only part where you need to write python is when you want to define custom actions. 
There's an excellent python library called `requests <http://docs.python-requests.org/en/master/>`_, which makes HTTP programming painless.
If Rasa just needs to interact with your other services over HTTP, your actions will all look 
something like this:


.. doctest::

   from rasa_core.actions import Action
   import requests

   class ApiAction(Action):
       def name(self):
           return "my_api_action"

       def run(self, dispatcher, tracker, domain):
           data = requests.get(url).json
           return [SlotSet("api_result", data)]

.. _custom_actions:

Defining Custom Actions
-----------------------


The easiest are ``UtterActions``, which just send a message to the user. You define them by adding an entry to the
action list that is named after the utterance. E.g. if there should be an action that utters the template called
``utter_greet`` you need to add ``utter_greet`` to the list of defined actions. In the above example yaml you can see that
all three of the defined actions are just named after utter templates and hence just respond with a message to
the user.

**What about more complicated actions?**
To continue with the restaurant example, if the user says "show me a Mexican restaurant",
your bot would execute the action ``ActionCheckRestaurants``, which might look like this:


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


Note that actions **do not mutate the tracker directly**.
Instead, an action can return ``events`` which are logged by the tracker and used to modify its 
own state.
