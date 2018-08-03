:desc: Understanding Rasa Core Domains
.. _domain:

Domain Format
=============

The ``Domain`` defines the universe in which your bot operates.
It specifies the ``intents``, ``entities``, ``slots``, and ``actions``
your bot should know about.
Optionally, it can also include ``templates`` for the things your bot can say.


As an example, the ``DefaultDomain`` has the following yaml definition:

.. literalinclude:: ../data/test_domains/default_with_slots.yml
   :language: yaml

**What does this mean?**

Your NLU model will define the ``intents`` and ``entities`` that you need to include
in the domain.

``slots`` are the things you want to keep track of during a conversation, see :ref:`slots` .

``actions`` are the things your bot can actually do.
For example, an ``action`` can:

* respond to a user
* make an external API call
* query a database

see :ref:`customactions`

For a more complete example domain, check the :doc:`quickstart`.


Custom Actions and Slots
^^^^^^^^^^^^^^^^^^^^^^^^

To reference custom actions and slots in your domain,
you need to reference them by their module path.
For example, if you have a module called ``my_actions`` containing
a class ``MyAwesomeAction``, and module ``my_slots`` containing ``MyAwesomeSlot``,
you would add these lines to the domain file:

.. code-block:: yaml

   actions:
     - my_actions.MyAwesomeAction
     ...

   slots:
     - my_slots.MyAwesomeSlot


.. _utter_templates:

Utterance templates
^^^^^^^^^^^^^^^^^^^

Utterance templates are messages the bot will send back to the user. Either
automatically by an action with the same name as the utterance (e.g. in the
above example the ``utter_default`` template and action) or by an action with
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
         dispatcher.utter_template("utter_default", tracker)
         # ... other code
         return []

If the template contains variables denoted with ``{my_variable}`` you can supply
values for the fields by passing them as key word arguments to ``utter_template``:

.. code-block:: python

  dispatcher.utter_template("utter_default", tracker, my_variable="my text")

Variations
----------

If you want to randomly vary the response sent to the user, you can list
multiple responses and Rasa will randomly pick one of them, e.g.:

.. code-block:: yaml

  templates:
    utter_greeting:
    - text: "Hey, {name}. How are you?"
    - text: "Hey, {name}. How is your day going?"

Ignoring entities for certain intents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want entities to be ignored for certain intents, you can add the ``use_entities: false``
parameter to the intent in your domain file like this:

.. code-block:: yaml

  intents:
    - greet: {use_entities: false}

This means that entities for those intents will be unfeaturized and therefore
will not impact the next action predictions. This is useful when you have
an intent where you don't care about the entities being picked up. If you list
your intents as normal without this parameter, the entities will be featurized as normal.

.. note::

    If you really want these entities not to influence action prediction we
    suggest you make the slots with the same name of type ``unfeaturized``
