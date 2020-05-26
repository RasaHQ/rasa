:desc: Define intents, entities, slots and actions in Rasa to build contextual a
       AI Assistants and chatbots using open source bot framework Rasa. a
 a
.. _domains: a
 a
Domains a
======= a
 a
.. edit-link:: a
 a
The ``Domain`` defines the universe in which your assistant operates. a
It specifies the ``intents``, ``entities``, ``slots``, and ``actions`` a
your bot should know about. Optionally, it can also include ``responses`` a
for the things your bot can say. a
 a
.. contents:: a
   :local: a
 a
 a
An example of a Domain a
---------------------- a
 a
As an example, the domain created by ``rasa init`` has the following yaml definition: a
 a
 a
.. literalinclude:: ../../rasa/cli/initial_project/domain.yml a
   :language: yaml a
 a
**What does this mean?** a
 a
Your NLU model will define the ``intents`` and ``entities`` that you a
need to include in the domain. The ``entities`` section lists all entities a
extracted by any :ref:`entity extractor<entity-extraction>` in your a
NLU pipeline. a
 a
For example: a
 a
.. code-block:: yaml a
 a
         entities: a
            - PERSON          # entity extracted by SpacyEntityExtractor a
            - time            # entity extracted by DucklingHTTPExtractor a
            - membership_type # custom entity extracted by CRFEntityExtractor a
            - priority        # custom entity extracted by CRFEntityExtractor a
 a
 a
:ref:`slots` hold information you want to keep track of during a conversation. a
A categorical slot called ``risk_level`` would be a
defined like this: a
 a
.. code-block:: yaml a
 a
         slots: a
            risk_level: a
               type: categorical a
               values: a
               - low a
               - medium a
               - high a
 a
 a
:ref:`Here <slot-classes>` you can find the full list of slot types defined by a
Rasa Core, along with syntax for including them in your domain file. a
 a
 a
:ref:`actions` are the things your bot can actually do. a
For example, an action could: a
 a
* respond to a user, a
* make an external API call, a
* query a database, or a
* just about anything! a
 a
Custom Actions and Slots a
------------------------ a
 a
To reference slots in your domain, you need to reference them by a
their **module path**. To reference custom actions, use their **name**. a
For example, if you have a module called ``my_actions`` containing a
a class ``MyAwesomeAction``, and module ``my_slots`` containing a
``MyAwesomeSlot``, you would add these lines to the domain file: a
 a
.. code-block:: yaml a
 a
   actions: a
     - my_custom_action a
     ... a
 a
   slots: a
     - my_slots.MyAwesomeSlot a
 a
 a
The ``name`` function of ``MyAwesomeAction`` needs to return a
``my_custom_action`` in this example (for more details, a
see :ref:`custom-actions`). a
 a
.. _domain-responses: a
 a
Responses a
--------- a
 a
Responses are messages the bot will send back to the user. There are a
two ways to use these responses: a
 a
1. If the name of the response starts with ``utter_``, the response can a
   directly be used as an action. You would add the response a
   to the domain: a
 a
   .. code-block:: yaml a
 a
      responses: a
        utter_greet: a
        - text: "Hey! How are you?" a
 a
   Afterwards, you can use the response as an action in the a
   stories: a
 a
   .. code-block:: story a
 a
      ## greet the user a
      * intent_greet a
        - utter_greet a
 a
   When ``utter_greet`` is run as an action, it will send the message from a
   the response back to the user. a
 a
2. You can use the responses to generate response messages from your a
   custom actions using the dispatcher: a
   ``dispatcher.utter_message(template="utter_greet")``. a
   This allows you to separate the logic of generating a
   the messages from the actual copy. In your custom action code, you can a
   send a message based on the response like this: a
 a
   .. code-block:: python a
 a
      from rasa_sdk.actions import Action a
 a
      class ActionGreet(Action): a
        def name(self): a
            return 'action_greet' a
 a
        def run(self, dispatcher, tracker, domain): a
            dispatcher.utter_message(template="utter_greet") a
            return [] a
 a
Images and Buttons a
------------------ a
 a
Responses defined in a domain's yaml file can contain images and a
buttons as well: a
 a
.. code-block:: yaml a
 a
   responses: a
     utter_greet: a
     - text: "Hey! How are you?" a
       buttons: a
       - title: "great" a
         payload: "great" a
       - title: "super sad" a
         payload: "super sad" a
     utter_cheer_up: a
     - text: "Here is something to cheer you up:" a
       image: "https://i.imgur.com/nGF1K8f.jpg" a
 a
.. note:: a
 a
   Please keep in mind that it is up to the implementation of the output a
   channel on how to display the defined buttons. The command line, for a
   example, can't display buttons or images, but tries to mimic them by a
   printing the options. a
 a
Custom Output Payloads a
---------------------- a
 a
You can also send any arbitrary output to the output channel using the a
``custom:`` key. Note that since the domain is in yaml format, the json a
payload should first be converted to yaml format. a
 a
For example, although date pickers are not a defined parameter in responses  a
because they are not supported by most channels, a Slack date picker a
can be sent like so: a
 a
.. code-block:: yaml a
 a
   responses: a
     utter_take_bet: a
     - custom: a
         blocks: a
         - type: section a
           text: a
             text: "Make a bet on when the world will end:" a
             type: mrkdwn a
           accessory: a
             type: datepicker a
             initial_date: '2019-05-21' a
             placeholder: a
               type: plain_text a
               text: Select a date a
 a
 a
Channel-Specific Responses a
-------------------------- a
 a
For each response, you can have multiple **response variations** (see :ref:`variations`). a
If you have certain response variations that you would like sent only to specific a
channels, you can specify this with the ``channel:`` key. The value should match a
the name defined in the ``name()`` method of the channel's ``OutputChannel`` a
class. Channel-specific responses are especially useful if creating custom a
output payloads that will only work in certain channels. a
 a
 a
.. code-block:: yaml a
 a
  responses: a
    utter_ask_game: a
    - text: "Which game would you like to play?" a
      channel: "slack" a
      custom: a
        - # payload for Slack dropdown menu to choose a game a
    - text: "Which game would you like to play?" a
      buttons: a
      - title: "Chess" a
        payload: '/inform{"game": "chess"}' a
      - title: "Checkers" a
        payload: '/inform{"game": "checkers"}' a
      - title: "Fortnite" a
        payload: '/inform{"game": "fortnite"}' a
 a
Each time your bot looks for responses, it will first check to see if there a
are any channel-specific response variations for the connected channel. If there are, it a
will choose **only** from these response variations. If no channel-specific response variations are a
found, it will choose from any response variations that do not have a defined ``channel``. a
Therefore, it is good practice to always have at least one response variation for each a
response that has no ``channel`` specified so that your bot can respond in all a
environments, including in the shell and in interactive learning. a
 a
Variables a
--------- a
 a
You can also use **variables** in your responses to insert information a
collected during the dialogue. You can either do that in your custom python a
code or by using the automatic slot filling mechanism. For example, if you a
have a response like this: a
 a
.. code-block:: yaml a
 a
  responses: a
    utter_greet: a
    - text: "Hey, {name}. How are you?" a
 a
Rasa will automatically fill that variable with a value found in a slot called a
``name``. a
 a
In custom code, you can retrieve a response by using: a
 a
.. testsetup:: a
 a
   from rasa_sdk.actions import Action a
 a
.. testcode:: a
 a
   class ActionCustom(Action): a
      def name(self): a
         return "action_custom" a
 a
      def run(self, dispatcher, tracker, domain): a
         # send utter default response to user a
         dispatcher.utter_message(template="utter_default") a
         # ... other code a
         return [] a
 a
If the response contains variables denoted with ``{my_variable}`` a
you can supply values for the fields by passing them as keyword a
arguments to ``utter_message``: a
 a
.. code-block:: python a
 a
  dispatcher.utter_message(template="utter_greet", my_variable="my text") a
 a
.. _variations: a
 a
Variations a
---------- a
 a
If you want to randomly vary the response sent to the user, you can list a
multiple **response variations** and Rasa will randomly pick one of them, e.g.: a
 a
.. code-block:: yaml a
 a
  responses: a
    utter_greeting: a
    - text: "Hey, {name}. How are you?" a
    - text: "Hey, {name}. How is your day going?" a
 a
.. _use_entities: a
 a
Ignoring entities for certain intents a
------------------------------------- a
 a
If you want all entities to be ignored for certain intents, you can a
add the ``use_entities: []`` parameter to the intent in your domain a
file like this: a
 a
.. code-block:: yaml a
 a
  intents: a
    - greet: a
        use_entities: [] a
 a
To ignore some entities or explicitly take only certain entities a
into account you can use this syntax: a
 a
.. code-block:: yaml a
 a
  intents: a
  - greet: a
      use_entities: a
        - name a
        - first_name a
      ignore_entities: a
        - location a
        - age a
 a
This means that excluded entities for those intents will be unfeaturized and therefore a
will not impact the next action predictions. This is useful when you have a
an intent where you don't care about the entities being picked up. If you list a
your intents as normal without this parameter, the entities will be a
featurized as normal. a
 a
.. note:: a
 a
    If you really want these entities not to influence action prediction we a
    suggest you make the slots with the same name of type ``unfeaturized``. a
 a
.. _session_config: a
 a
Session configuration a
--------------------- a
 a
A conversation session represents the dialogue between the assistant and the user. a
Conversation sessions can begin in three ways: a
 a
  1. the user begins the conversation with the assistant, a
  2. the user sends their first message after a configurable period of inactivity, or a
  3. a manual session start is triggered with the ``/session_start`` intent message. a
 a
You can define the period of inactivity after which a new conversation a
session is triggered in the domain under the ``session_config`` key. a
``session_expiration_time`` defines the time of inactivity in minutes after which a a
new session will begin. ``carry_over_slots_to_new_session`` determines whether a
existing set slots should be carried over to new sessions. a
 a
The default session configuration looks as follows: a
 a
.. code-block:: yaml a
 a
  session_config: a
    session_expiration_time: 60  # value in minutes, 0 means infinitely long a
    carry_over_slots_to_new_session: true  # set to false to forget slots between sessions a
 a
This means that if a user sends their first message after 60 minutes of inactivity, a a
new conversation session is triggered, and that any existing slots are carried over a
into the new session. Setting the value of ``session_expiration_time`` to 0 means a
that sessions will not end (note that the ``action_session_start`` action will still a
be triggered at the very beginning of conversations). a
 a
.. note:: a
 a
  A session start triggers the default action ``action_session_start``. Its default a
  implementation moves all existing slots into the new session. Note that all a
  conversations begin with an ``action_session_start``. Overriding this action could a
  for instance be used to initialise the tracker with slots from an external API a
  call, or to start the conversation with a bot message. The docs on a
  :ref:`custom_session_start` shows you how to do that. a
 a