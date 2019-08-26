:desc: Define intents, entities, slots and actions in Rasa to build contextual
       AI Assistants and chatbots using open source bot framework Rasa.

.. _domains:

Domains
=======

.. edit-link::

The ``Domain`` defines the universe in which your assistant operates.
It specifies the ``intents``, ``entities``, ``slots``, and ``actions``
your bot should know about. Optionally, it can also include ``templates``
for the things your bot can say.

.. contents::
   :local:


An example of a Domain
----------------------

As an example, the ``DefaultDomain`` has the following yaml definition:


.. literalinclude:: ../../rasa/cli/initial_project/domain.yml
   :language: yaml

**What does this mean?**

Your NLU model will define the ``intents`` and ``entities`` that you
need to include in the domain.

:ref:`slots` hold information you want to keep track of during a conversation.
A categorical slot called ``risk_level`` would be
defined like this:

.. code-block:: yaml

         slots:
            risk_level:
               type: categorical
               values:
               - low
               - medium
               - high


:ref:`Here <slot-classes>` you can find the full list of slot types defined by
Rasa Core, along with syntax for including them in your domain file.


:ref:`actions` are the things your bot can actually do.
For example, an action could:

* respond to a user,
* make an external API call,
* query a database, or
* just about anything!

Custom Actions and Slots
------------------------

To reference slots in your domain, you need to reference them by
their **module path**. To reference custom actions, use their **name**.
For example, if you have a module called ``my_actions`` containing
a class ``MyAwesomeAction``, and module ``my_slots`` containing
``MyAwesomeSlot``, you would add these lines to the domain file:

.. code-block:: yaml

   actions:
     - my_custom_action
     ...

   slots:
     - my_slots.MyAwesomeSlot


The ``name`` function of ``MyAwesomeAction`` needs to return
``my_custom_action`` in this example (for more details,
see :ref:`custom-actions`).

.. _utter_templates:

Utterance templates
-------------------

Utterance templates are messages the bot will send back to the user. There are
two ways to use these templates:

1. If the name of the template starts with ``utter_``, the utterance can
   directly be used as an action. You would add the utterance template
   to the domain:

   .. code-block:: yaml

      templates:
        utter_greet:
        - text: "Hey! How are you?"

   Afterwards, you can use the template as an action in the
   stories:

   .. code-block:: story

      ## greet the user
      * intent_greet
        - utter_greet

   When ``utter_greet`` is run as an action, it will send the message from
   the template back to the user.

2. You can use the templates to generate response messages from your
   custom actions using the dispatcher:
   ``dispatcher.utter_template("utter_greet", tracker)``.
   This allows you to separate the logic of generating
   the messages from the actual copy. In you custom action code, you can
   send a message based on the template like this:

   .. code-block:: python

      from rasa_sdk.actions import Action

      class ActionGreet(Action):
        def name(self):
            return 'action_greet'

        def run(self, dispatcher, tracker, domain):
            dispatcher.utter_template("utter_greet", tracker)
            return []

Images and Buttons
------------------

Templates defined in a domain's yaml file can contain images and
buttons as well:

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
       image: "https://i.imgur.com/nGF1K8f.jpg"

.. note::

   Please keep in mind that it is up to the implementation of the output
   channel on how to display the defined buttons. The command line, for
   example, can't display buttons or images, but tries to mimic them by
   printing the options.

Custom Output Payloads
----------------------

You can also send any arbitrary output to the output channel using the
``custom:`` key. Note that since the domain is in yaml format, the json
payload should first be converted to yaml format.

For example, although date pickers are not a defined parameter in utterance
templates because they are not supported by most channels, a Slack date picker
can be sent like so:

.. code-block:: yaml

   templates:
     utter_take_bet:
     - custom:
         blocks:
         - type: section
           text:
             text: "Make a bet on when the world will end:"
             type: mrkdwn
           accessory:
             type: datepicker
             initial_date: '2019-05-21'
             placeholder:
               type: plain_text
               text: Select a date


Channel-Specific Utterances
---------------------------

If you have certain utterances that you would like sent only to specific
channels, you can specify this with the ``channel:`` key. The value should match
the name defined in the ``name()`` method of the channel's ``OutputChannel``
class. Channel-specific utterances are especially useful if creating custom
output payloads that will only work in certain channels.


.. code-block:: yaml

  templates:
    utter_ask_game:
    - text: "Which game would you like to play?"
      channel: "slack"
      custom:
        - # payload for Slack dropdown menu to choose a game
    - text: "Which game would you like to play?"
      buttons:
      - title: "Chess"
        payload: '/inform{"game": "chess"}'
      - title: "Checkers"
        payload: '/inform{"game": "checkers"}'
      - title: "Fortnite"
        payload: '/inform{"game": "fortnite"}'

Each time your bot looks for utterances, it will first check to see if there
are any channel-specific templates for the connected channel. If there are, it
will choose **only** from these utterances. If no channel-specific templates are
found, it will choose from any utterances that do not have a defined ``channel``.
Therefore, it is good practice to always have at least one template for each
utterance that has no ``channel`` specified so that your bot can respond in all
environments, including in the shell and in interactive learning.

Variables
---------

You can also use **variables** in your templates to insert information
collected during the dialogue. You can either do that in your custom python
code or by using the automatic slot filling mechanism. For example, if you
have a template like this:

.. code-block:: yaml

  templates:
    utter_greet:
    - text: "Hey, {name}. How are you?"

Rasa will automatically fill that variable with a value found in a slot called
``name``.

In custom code, you can retrieve a template by using:

.. testsetup::

   from rasa_sdk.actions import Action

.. testcode::

   class ActionCustom(Action):
      def name(self):
         return "action_custom"

      def run(self, dispatcher, tracker, domain):
         # send utter default template to user
         dispatcher.utter_template("utter_default", tracker)
         # ... other code
         return []

If the template contains variables denoted with ``{my_variable}``
you can supply values for the fields by passing them as keyword
arguments to ``utter_template``:

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

.. _use_entities:

Ignoring entities for certain intents
-------------------------------------

If you want all entities to be ignored for certain intents, you can
add the ``use_entities: []`` parameter to the intent in your domain
file like this:

.. code-block:: yaml

  intents:
    - greet:
        use_entities: []

To ignore some entities or explicitly take only certain entities
into account you can use this syntax:

.. code-block:: yaml

  intents:
  - greet:
      use_entities: 
        - name
        - first_name
      ignore_entities:
        - location
        - age

This means that excluded entities for those intents will be unfeaturized and therefore
will not impact the next action predictions. This is useful when you have
an intent where you don't care about the entities being picked up. If you list
your intents as normal without this parameter, the entities will be
featurized as normal.

.. note::

    If you really want these entities not to influence action prediction we
    suggest you make the slots with the same name of type ``unfeaturized``.
