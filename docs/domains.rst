.. _domain:

Domain, Slots, and Actions
==========================

Domain
^^^^^^

The ``Domain`` defines the universe in which your bot operates. It specifies exactly:

* which ``intents`` you are expecting to respond to
* which ``slots`` you wish to track
* which ``actions`` your bot can take

For example, the ``DefaultDomain`` has the following yaml definition:

.. literalinclude:: ../data/test_domains/default_with_slots.yml
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

.. note::

  For mor information about the utter template format (e.g. the use of
  variables like ``{name}`` or buttons) take a look at :ref:`utter_templates`.

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


Putting it all together
-----------------------

Let's add just this one new action to a custom domain (assuming we stored the
action in a module called ``restaurant.actions``):

.. code-block:: yaml

    actions:
      - utter_default
      - utter_greet
      - utter_goodbye
      - restaurant.actions.ActionCheckRestaurants   # custom action


We only show the changed action list here, you also need to include the other
parts from the original domain! The point of this is just to show how the pieces
fit together. As you can see, in the ``actions`` section
of your domain, you can list utter actions (which respond an utter template to the user) as well as custom
actions using their module path.

For an example you can run, check the :doc:`tutorial_basics`.

.. _utter_templates:

Utterance templates
^^^^^^^^^^^^^^^^^^^

Utterance templates are messages the bot will send back to the user. Either
automatically by an action with the same name as the utterance (e.g. in the
above example the `utter_default` template and action) or by an action with
custom code.

Images and Buttons
------------------

Templates defined in a domains yaml file can contain images and buttons as well:

.. code-block:: yaml

   templates:
     utter_greet:
     - text: "Hey! How are you?"
       buttons:
       - title: "great"
         payload: "great"
       - title: "super sad"
         payload: "super sad"
     utter_cheer_up:
     - text: "Here is something to cheer you up:"
       image: "https://cdn77.eatliver.com/wp-content/uploads/2017/10/trump-frog.jpg"

.. note::

   Please keep in mind that it is up to the implementation of the output
   channel on how to display the defined buttons. E.g. the cmdline
   interface can not display buttons or images, but tries to mimic them in
   the command line.

Variables
---------

You can also use **variables** in your templates to insert information
collected during the dialogue. You can either do that in your custom python
code or by using the automatic slot filling mechanism. E.g if you got a template
like this:

.. code-block:: yaml

  templates:
    utter_greet:
    - text: "Hey, {name}. How are you?"

Rasa will automatically fill that variable with a value found in a slot called
``name``.

In custom code, you can retrieve a template by using:

.. testsetup::

   from rasa_core.actions import Action

.. testcode::

   class ActionCustom(Action):
      def name(self):
         return "action_custom"

      def run(self, dispatcher, tracker, domain):
         # send utter default template to user
         dispatcher.utter_template("utter_default")
         # ... other code
         return []

If the template contains variables denoted with ``{my_variable}`` you can supply
values for the fields by passing them as key word arguments to ``utter_template``:

.. code-block:: python

  dispatcher.utter_template("utter_default", my_variable="my text")

Variations
----------

If you want to randomly vary the response send to the user, you can list
multiple responses and the bot will randomly pick one of them, e.g.:

.. code-block:: yaml

  templates:
    utter_greeting:
    - text: "Hey, {name}. How are you?"
    - text: "Hey, {name}. How is your day going?"


.. _slot_types:

Slots
^^^^^

Most slots influence the prediction of the next action the bot should run. For the
prediction, the slots value is not used directly, but rather it is featurized.
E.g. for a slot of type ``text``, the value is irrelevant, for the featurization
the only thing that matters is if a text is set or not. In a slot of
type ``unfeaturized`` any value can be stored, the slot never influences the
prediction.

The **choice of a slots type should be done with care**. If a slots value should
influence the dialogue flow (e.g. the users age influences which
question follows next) you should choose a slot where the value influences
the dialogue model.

These are all of the predefined slot classes and what they're useful for:

.. option:: text

   :Use For: User preferences where you only care whether or not they've
             been specified.
   :Example:
      .. sourcecode:: yaml

         slots:
            cuisine:
               type: text
   :Description:
       Results in the feature of the slot being set to ``1`` if any value is set.
       Otherwise the feature will be set to ``0`` (no value is set).


.. option:: bool

   :Use For: True or False
   :Example:
      .. sourcecode:: yaml

         slots:
            is_authenticated:
               type: bool
   :Description:
       Checks if slot is set and if True


.. option:: categorical

   :Use For: Slots which can take one of N values
   :Example:
      .. sourcecode:: yaml

         slots:
            risc_level:
               type: categorical
               values:
               - low
               - medium
               - high

   :Description:
      Creates a one-hot encoding describing which of the ``values`` matched.


.. option:: float

   :Use For: Continuous values
   :Example:
      .. sourcecode:: yaml

         slots:
            temperature:
               type: float
               min_value: -100.0
               max_value:  100.0

   :Defaults: ``max_value=1.0``, ``min_value=0.0``
   :Description:
      All values below ``min_value`` will be treated as ``min_value``, the same
      happens for values above ``max_value``. Hence, if ``max_value`` is set to
      ``1``, there is no difference between the slot values ``2`` and ``3.5`` in
      terms of featurization (e.g. both values will influence the dialogue in
      the same way and the model can not learn to differentiate between them).


.. option:: list

   :Use For: Lists of values
   :Example:
      .. sourcecode:: yaml

         slots:
            shopping_items:
               type: list
   :Description:
       The feature of this slot is set to ``1`` if a value with a list is set,
       where the list is not empty. If no value is set, or the empty list is the
       set value, the feature will be ``0``. The **length of the list stored in
       the slot does not influence the dialogue**.


.. option:: unfeaturized

   :Use For: Data you want to store which shouldn't influence the dialogue flow
   :Example:
      .. sourcecode:: yaml

         slots:
            internal_user_id:
               type: unfeaturized
   :Description:
       There will not be any featurization of this slot, hence its value does
       not influence the dialogue flow and is ignored when predicting the next
       action the bot should run.


.. option:: data

   :Use For:  Base class for creating own slots
   :Example:
      .. warning:: This type should not be used directly, but rather be subclassed.

   :Description:
      User has to subclass this and define the ``as_feature`` method containing
      any custom logic.
