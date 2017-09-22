Domains and Actions
===================


The ``Domain`` defines the universe in which your bot operates. It specifies exactly:

* which ``intents`` you are expecting to respond to
* which ``slots`` you wish to track
* which ``actions`` your bot can take

For example, the ``DefaultDomain`` has the following yaml definition (no slots here - but we will get there):

.. literalinclude:: ../examples/default_domain.yml
   :language: yaml

**What does this mean?**

An ``intent`` is a string like ``"greet"`` or ``"restaurant_search"``.
It describes what your user *probably meant to say*. 
For example, "show me Mexican restaurants", and "I want to have lunch"
could both be described as a ``restaurant_search`` intent. 


``slots`` are the things you want to keep track of during a conversation.
For example, in the messages above you would want to store "Mexican" as a cuisine type.
The tracker has an attribute like ``tracker.get_slot("cuisine")`` which will return ``"Mexican"``

``actions`` are the things your bot can actually do.
They are invoked by calling the ``action.run()`` method.
For example, an ``action`` can:

* respond to a user
* make an external API call
* query a database

Defining Custom Actions
-----------------------


The easiest are ``UtterActions``, which just send a message to the user. You define them by adding an entry to the
action list that is named after the utterance. E.g. if there should be an action that utters the template called
``utter_greet`` you need to add ``greet`` to the list of defined actions. In the above example yaml you can see that
all three of the defined actions are just named after utter templates and hence just respond with a message to
the user.

**What about more complicated actions?**
To continue with the restaurant example, if the user says "show me a Mexican restaurant",
your bot would execute the action ``ActionCheckRestaurants``, which might look like this:


.. testcode::

   from rasa_core.actions import Action
   from rasa_core.events import SetSlot

   class ActionCheckRestaurants(Action):
      def name(self):
         return "check_restaurants"

      def run(self, dispatcher, tracker, domain):
         cuisine = tracker.get_slot('cuisine')
         q = "select * from restaurants where cuisine='{0}' limit 1".format(cuisine)
         result = db.query(q)

         return [SetSlot("matches", result if result is not None else [])]


Note that actions **do not mutate the tracker directly**.
Instead, an action can return ``events`` which are logged by the tracker and used to modify its 
own state.


Putting it all together
-----------------------

Let's add just this one new action to a custom domain (assuming we stored the
action in a module called ``restaurant.actions``):

.. code-block:: yaml

   intents:
      - greet
      - default
      - goodbye

   entities:
      - name

   templates:
      utter_greet:
         - "hey there!"
      utter_goodbye:
         - "goodbye :("
      utter_default:
         - "default message"

   actions:
      - default
      - greet
      - goodbye
      - restaurant.actions.ActionCheckRestaurants


The point of this is just to show how the pieces fit together. As you can see, in the ``actions`` section
of your domain, you can list utter actions (which respond an utter template to the user) as well as custom
actions using their module path.

For an example you can run, check the :doc:`tutorial_scratch`.