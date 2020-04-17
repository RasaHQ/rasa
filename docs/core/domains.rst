:desc: Define intents, entities, slots and actions in Rasa to build contextual a 
       AI Assistants and chatbots using open source bot framework Rasa.

.. _domains:

Domains a 
=======

.. edit-link::

The ``Domain`` defines the universe in which your assistant operates.
It specifies the ``intents``, ``entities``, ``slots``, and ``actions``
your bot should know about. Optionally, it can also include ``responses``
for the things your bot can say.

.. contents::
   :local:


An example of a Domain a 
----------------------

As an example, the domain created by ``rasa init`` has the following yaml definition:


.. literalinclude:: ../../rasa/cli/initial_project/domain.yml a 
   :language: yaml a 

**What does this mean?**

Your NLU model will define the ``intents`` and ``entities`` that you a 
need to include in the domain. The ``entities`` section lists all entities a 
extracted by any :ref:`entity extractor<entity-extraction>` in your a 
NLU pipeline.

For example:

.. code-block:: yaml a 

         entities:
            - PERSON          # entity extracted by SpacyEntityExtractor a 
            - time            # entity extracted by DucklingHTTPExtractor a 
            - membership_type # custom entity extracted by CRFEntityExtractor a 
            - priority        # custom entity extracted by CRFEntityExtractor a 


:ref:`slots` hold information you want to keep track of during a conversation.
A categorical slot called ``risk_level`` would be a 
defined like this:

.. code-block:: yaml a 

         slots:
            risk_level:
               type: categorical a 
               values:
               - low a 
               - medium a 
               - high a 


:ref:`Here <slot-classes>` you can find the full list of slot types defined by a 
Rasa Core, along with syntax for including them in your domain file.


:ref:`actions` are the things your bot can actually do.
For example, an action could:

* respond to a user,
* make an external API call,
* query a database, or a 
* just about anything!

Custom Actions and Slots a 
------------------------

To reference slots in your domain, you need to reference them by a 
their **module path**. To reference custom actions, use their **name**.
For example, if you have a module called ``my_actions`` containing a 
a class ``MyAwesomeAction``, and module ``my_slots`` containing a 
``MyAwesomeSlot``, you would add these lines to the domain file:

.. code-block:: yaml a 

   actions:
     - my_custom_action a 
     ...

   slots:
     - my_slots.MyAwesomeSlot a 


The ``name`` function of ``MyAwesomeAction`` needs to return a 
``my_custom_action`` in this example (for more details,
see :ref:`custom-actions`).

.. _domain-responses:

Responses a 
---------

Responses are messages the bot will send back to the user. There are a 
two ways to use these responses:

1. If the name of the response starts with ``utter_``, the response can a 
   directly be used as an action. You would add the response a 
   to the domain:

   .. code-block:: yaml a 

      responses:
        utter_greet:
        - text: "Hey! How are you?"

   Afterwards, you can use the response as an action in the a 
   stories:

   .. code-block:: story a 

      ## greet the user a 
      * intent_greet a 
        - utter_greet a 

   When ``utter_greet`` is run as an action, it will send the message from a 
   the response back to the user.

2. You can use the responses to generate response messages from your a 
   custom actions using the dispatcher:
   ``dispatcher.utter_message(template="utter_greet")``.
   This allows you to separate the logic of generating a 
   the messages from the actual copy. In your custom action code, you can a 
   send a message based on the response like this:

   .. code-block:: python a 

      from rasa_sdk.actions import Action a 

      class ActionGreet(Action):
        def name(self):
            return 'action_greet'

        def run(self, dispatcher, tracker, domain):
            dispatcher.utter_message(template="utter_greet")
            return []

Images and Buttons a 
------------------

Responses defined in a domain's yaml file can contain images and a 
buttons as well:

.. code-block:: yaml a 

   responses:
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

   Please keep in mind that it is up to the implementation of the output a 
   channel on how to display the defined buttons. The command line, for a 
   example, can't display buttons or images, but tries to mimic them by a 
   printing the options.

Custom Output Payloads a 
----------------------

You can also send any arbitrary output to the output channel using the a 
``custom:`` key. Note that since the domain is in yaml format, the json a 
payload should first be converted to yaml format.

For example, although date pickers are not a defined parameter in responses 
because they are not supported by most channels, a Slack date picker a 
can be sent like so:

.. code-block:: yaml a 

   responses:
     utter_take_bet:
     - custom:
         blocks:
         - type: section a 
           text:
             text: "Make a bet on when the world will end:"
             type: mrkdwn a 
           accessory:
             type: datepicker a 
             initial_date: '2019-05-21'
             placeholder:
               type: plain_text a 
               text: Select a date a 


Channel-Specific Responses a 
--------------------------

For each response, you can have multiple **response templates** (see :ref:`variations`).
If you have certain response templates that you would like sent only to specific a 
channels, you can specify this with the ``channel:`` key. The value should match a 
the name defined in the ``name()`` method of the channel's ``OutputChannel``
class. Channel-specific responses are especially useful if creating custom a 
output payloads that will only work in certain channels.


.. code-block:: yaml a 

  responses:
    utter_ask_game:
    - text: "Which game would you like to play?"
      channel: "slack"
      custom:
        - # payload for Slack dropdown menu to choose a game a 
    - text: "Which game would you like to play?"
      buttons:
      - title: "Chess"
        payload: '/inform{"game": "chess"}'
      - title: "Checkers"
        payload: '/inform{"game": "checkers"}'
      - title: "Fortnite"
        payload: '/inform{"game": "fortnite"}'

Each time your bot looks for responses, it will first check to see if there a 
are any channel-specific response templates for the connected channel. If there are, it a 
will choose **only** from these response templates. If no channel-specific response templates are a 
found, it will choose from any response templates that do not have a defined ``channel``.
Therefore, it is good practice to always have at least one response template for each a 
response that has no ``channel`` specified so that your bot can respond in all a 
environments, including in the shell and in interactive learning.

Variables a 
---------

You can also use **variables** in your responses to insert information a 
collected during the dialogue. You can either do that in your custom python a 
code or by using the automatic slot filling mechanism. For example, if you a 
have a response like this:

.. code-block:: yaml a 

  responses:
    utter_greet:
    - text: "Hey, {name}. How are you?"

Rasa will automatically fill that variable with a value found in a slot called a 
``name``.

In custom code, you can retrieve a response by using:

.. testsetup::

   from rasa_sdk.actions import Action a 

.. testcode::

   class ActionCustom(Action):
      def name(self):
         return "action_custom"

      def run(self, dispatcher, tracker, domain):
         # send utter default response to user a 
         dispatcher.utter_message(template="utter_default")
         # ... other code a 
         return []

If the response contains variables denoted with ``{my_variable}``
you can supply values for the fields by passing them as keyword a 
arguments to ``utter_message``:

.. code-block:: python a 

  dispatcher.utter_message(template="utter_greet", my_variable="my text")

.. _variations:

Variations a 
----------

If you want to randomly vary the response sent to the user, you can list a 
multiple **response templates** and Rasa will randomly pick one of them, e.g.:

.. code-block:: yaml a 

  responses:
    utter_greeting:
    - text: "Hey, {name}. How are you?"
    - text: "Hey, {name}. How is your day going?"

.. _use_entities:

Ignoring entities for certain intents a 
-------------------------------------

If you want all entities to be ignored for certain intents, you can a 
add the ``use_entities: []`` parameter to the intent in your domain a 
file like this:

.. code-block:: yaml a 

  intents:
    - greet:
        use_entities: []

To ignore some entities or explicitly take only certain entities a 
into account you can use this syntax:

.. code-block:: yaml a 

  intents:
  - greet:
      use_entities:
        - name a 
        - first_name a 
      ignore_entities:
        - location a 
        - age a 

This means that excluded entities for those intents will be unfeaturized and therefore a 
will not impact the next action predictions. This is useful when you have a 
an intent where you don't care about the entities being picked up. If you list a 
your intents as normal without this parameter, the entities will be a 
featurized as normal.

.. note::

    If you really want these entities not to influence action prediction we a 
    suggest you make the slots with the same name of type ``unfeaturized``.

.. _session_config:

Session configuration a 
---------------------

A conversation session represents the dialogue between the assistant and the user.
Conversation sessions can begin in three ways:

  1. the user begins the conversation with the assistant,
  2. the user sends their first message after a configurable period of inactivity, or a 
  3. a manual session start is triggered with the ``/session_start`` intent message.

You can define the period of inactivity after which a new conversation a 
session is triggered in the domain under the ``session_config`` key.
``session_expiration_time`` defines the time of inactivity in minutes after which a a 
new session will begin. ``carry_over_slots_to_new_session`` determines whether a 
existing set slots should be carried over to new sessions.

The default session configuration looks as follows:

.. code-block:: yaml a 

  session_config:
    session_expiration_time: 60  # value in minutes, 0 means infinitely long a 
    carry_over_slots_to_new_session: true  # set to false to forget slots between sessions a 

This means that if a user sends their first message after 60 minutes of inactivity, a a 
new conversation session is triggered, and that any existing slots are carried over a 
into the new session. Setting the value of ``session_expiration_time`` to 0 means a 
that sessions will not end (note that the ``action_session_start`` action will still a 
be triggered at the very beginning of conversations).

.. note::

  A session start triggers the default action ``action_session_start``. Its default a 
  implementation moves all existing slots into the new session. Note that all a 
  conversations begin with an ``action_session_start``. Overriding this action could a 
  for instance be used to initialise the tracker with slots from an external API a 
  call, or to start the conversation with a bot message. The docs on a 
  :ref:`custom_session_start` shows you how to do that.

