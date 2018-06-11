.. _domain:

Domain Format
=============

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

  For more information about the utter template format (e.g. the use of
  variables like ``{name}`` or buttons) take a look at :ref:`utter_templates`.



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

If we want to use our new action for a specific story, we only have to add the canonical name of the action to a story. 
In our case we would have to add ``- action_check_restaurants`` to a story in our ``stories.md``.

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


