:desc: Store information the user provided as well as information from database a
       queries in slots to influence how the machine learning based dialogue a
       continues. a
 a
.. _slots: a
 a
Slots a
===== a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
What are slots? a
--------------- a
 a
**Slots are your bot's memory.** They act as a key-value store a
which can be used to store information the user provided (e.g their home city) a
as well as information gathered about the outside world (e.g. the result of a a
database query). a
 a
Most of the time, you want slots to influence how the dialogue progresses. a
There are different slot types for different behaviors. a
 a
For example, if your user has provided their home city, you might a
have a ``text`` slot called ``home_city``. If the user asks for the a
weather, and you *don't* know their home city, you will have to ask a
them for it. A ``text`` slot only tells Rasa Core whether the slot a
has a value. The specific value of a ``text`` slot (e.g. Bangalore a
or New York or Hong Kong) doesn't make any difference. a
 a
If the value itself is important, use a ``categorical`` or a ``bool`` slot. a
There are also ``float``, and ``list`` slots. a
If you just want to store some data, but don't want it to affect the flow a
of the conversation, use an ``unfeaturized`` slot. a
 a
 a
How Rasa Uses Slots a
------------------- a
 a
The ``Policy`` doesn't have access to the a
value of your slots. It receives a featurized representation. a
As mentioned above, for a ``text`` slot the value is irrelevant. a
The policy just sees a ``1`` or ``0`` depending on whether it is set. a
 a
**You should choose your slot types carefully!** a
 a
How Slots Get Set a
----------------- a
 a
You can provide an initial value for a slot in your domain file: a
 a
.. code-block:: yaml a
 a
    slots: a
      name: a
        type: text a
        initial_value: "human" a
 a
You can get the value of a slot using ``.get_slot()`` inside ``actions.py`` for example:   a
 a
.. code-block:: python a
 a
       data = tracker.get_slot("slot-name") a
 a
 a
 a
There are multiple ways that slots are set during a conversation: a
 a
Slots Set from NLU a
~~~~~~~~~~~~~~~~~~ a
 a
If your NLU model picks up an entity, and your domain contains a a
slot with the same name, the slot will be set automatically. For example: a
 a
.. code-block:: story a
 a
   # story_01 a
   * greet{"name": "Ali"} a
     - slot{"name": "Ali"} a
     - utter_greet a
 a
In this case, you don't have to include the ``- slot{}`` part in the a
story, because it is automatically picked up. a
 a
To disable this behavior for a particular slot, you can set the a
``auto_fill`` attribute to ``False`` in the domain file: a
 a
.. code-block:: yaml a
     a
    slots: a
      name: a
        type: text a
        auto_fill: False a
 a
 a
Slots Set By Clicking Buttons a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
You can use buttons as a shortcut. a
Rasa Core will send messages starting with a ``/`` to the a
``RegexInterpreter``, which expects NLU input in the same format a
as in story files, e.g. ``/intent{entities}``. For example, if you let a
users choose a color by clicking a button, the button payloads might a
be ``/choose{"color": "blue"}`` and ``/choose{"color": "red"}``. a
 a
You can specify this in your domain file like this: a
(see details in :ref:`domains`) a
 a
.. code-block:: yaml a
 a
  utter_ask_color: a
  - text: "what color would you like?" a
    buttons: a
    - title: "blue" a
      payload: '/choose{"color": "blue"}' a
    - title: "red" a
      payload: '/choose{"color": "red"}' a
 a
 a
Slots Set by Actions a
~~~~~~~~~~~~~~~~~~~~ a
 a
The second option is to set slots by returning events in :ref:`custom actions <custom-actions>`. a
In this case, your stories need to include the slots. a
For example, you have a custom action to fetch a user's profile, and a
you have a ``categorical`` slot called ``account_type``. a
When the ``fetch_profile`` action is run, it returns a a
:class:`rasa.core.events.SlotSet` event: a
 a
.. code-block:: yaml a
 a
   slots: a
      account_type: a
         type: categorical a
         values: a
         - premium a
         - basic a
 a
.. code-block:: python a
 a
   from rasa_sdk.actions import Action a
   from rasa_sdk.events import SlotSet a
   import requests a
 a
   class FetchProfileAction(Action): a
       def name(self): a
           return "fetch_profile" a
 a
       def run(self, dispatcher, tracker, domain): a
           url = "http://myprofileurl.com" a
           data = requests.get(url).json a
           return [SlotSet("account_type", data["account_type"])] a
 a
 a
.. code-block:: story a
 a
   # story_01 a
   * greet a
     - action_fetch_profile a
     - slot{"account_type" : "premium"} a
     - utter_welcome_premium a
 a
   # story_02 a
   * greet a
     - action_fetch_profile a
     - slot{"account_type" : "basic"} a
     - utter_welcome_basic a
 a
 a
In this case you **do** have to include the ``- slot{}`` part in your stories. a
Rasa Core will learn to use this information to decide on the correct action to a
take (in this case, ``utter_welcome_premium`` or ``utter_welcome_basic``). a
 a
.. note:: a
   It is **very easy** to forget about slots if you are writing a
   stories by hand. We strongly recommend that you build up these a
   stories using :ref:`section_interactive_learning_forms` rather than writing them. a
 a
 a
.. _slot-classes: a
 a
Slot Types a
---------- a
 a
Text Slot a
~~~~~~~~~ a
 a
.. option:: text a
 a
  :Use For: User preferences where you only care whether or not they've a
            been specified. a
  :Example: a
     .. sourcecode:: yaml a
 a
        slots: a
           cuisine: a
              type: text a
  :Description: a
      Results in the feature of the slot being set to ``1`` if any value is set. a
      Otherwise the feature will be set to ``0`` (no value is set). a
 a
Boolean Slot a
~~~~~~~~~~~~ a
 a
.. option:: bool a
 a
  :Use For: True or False a
  :Example: a
     .. sourcecode:: yaml a
 a
        slots: a
           is_authenticated: a
              type: bool a
  :Description: a
      Checks if slot is set and if True a
 a
Categorical Slot a
~~~~~~~~~~~~~~~~ a
 a
.. option:: categorical a
 a
  :Use For: Slots which can take one of N values a
  :Example: a
     .. sourcecode:: yaml a
 a
        slots: a
           risk_level: a
              type: categorical a
              values: a
              - low a
              - medium a
              - high a
 a
  :Description: a
     Creates a one-hot encoding describing which of the ``values`` matched. a
     A default value ``__other__`` is automatically added to the user-defined a
     values. All values encountered which are not explicitly defined in the  a
     domain are mapped to ``__other__`` for featurization. The value  a
     ``__other__`` should not be used as a user-defined value; if it is, it  a
     will still behave as the default to which all unseen values are mapped. a
 a
Float Slot a
~~~~~~~~~~ a
 a
.. option:: float a
 a
  :Use For: Continuous values a
  :Example: a
     .. sourcecode:: yaml a
 a
        slots: a
           temperature: a
              type: float a
              min_value: -100.0 a
              max_value:  100.0 a
 a
  :Defaults: ``max_value=1.0``, ``min_value=0.0`` a
  :Description: a
     All values below ``min_value`` will be treated as ``min_value``, the same a
     happens for values above ``max_value``. Hence, if ``max_value`` is set to a
     ``1``, there is no difference between the slot values ``2`` and ``3.5`` in a
     terms of featurization (e.g. both values will influence the dialogue in a
     the same way and the model can not learn to differentiate between them). a
 a
List Slot a
~~~~~~~~~ a
 a
.. option:: list a
 a
  :Use For: Lists of values a
  :Example: a
     .. sourcecode:: yaml a
 a
        slots: a
           shopping_items: a
              type: list a
  :Description: a
      The feature of this slot is set to ``1`` if a value with a list is set, a
      where the list is not empty. If no value is set, or the empty list is the a
      set value, the feature will be ``0``. The **length of the list stored in a
      the slot does not influence the dialogue**. a
 a
.. _unfeaturized-slot: a
 a
Unfeaturized Slot a
~~~~~~~~~~~~~~~~~ a
 a
.. option:: unfeaturized a
 a
  :Use For: Data you want to store which shouldn't influence the dialogue flow a
  :Example: a
     .. sourcecode:: yaml a
 a
        slots: a
           internal_user_id: a
              type: unfeaturized a
  :Description: a
      There will not be any featurization of this slot, hence its value does a
      not influence the dialogue flow and is ignored when predicting the next a
      action the bot should run. a
 a
Custom Slot Types a
----------------- a
 a
Maybe your restaurant booking system can only handle bookings a
for up to 6 people. In this case you want the *value* of the a
slot to influence the next selected action (and not just whether a
it's been specified). You can do this by defining a custom slot class. a
 a
In the code below, we define a slot class called ``NumberOfPeopleSlot``. a
The featurization defines how the value of this slot gets converted to a vector a
to our machine learning model can deal with. a
Our slot has three possible "values", which we can represent with a
a vector of length ``2``. a
 a
+---------------+------------------------------------------+ a
| ``(0,0)``     | not yet set                              | a
+---------------+------------------------------------------+ a
| ``(1,0)``     | between 1 and 6                          | a
+---------------+------------------------------------------+ a
| ``(0,1)``     | more than 6                              | a
+---------------+------------------------------------------+ a
 a
 a
.. testcode:: a
 a
   from rasa.core.slots import Slot a
 a
   class NumberOfPeopleSlot(Slot): a
 a
       def feature_dimensionality(self): a
           return 2 a
 a
       def as_feature(self): a
           r = [0.0] * self.feature_dimensionality() a
           if self.value: a
               if self.value <= 6: a
                   r[0] = 1.0 a
               else: a
                   r[1] = 1.0 a
           return r a
 a
Now we also need some training stories, so that Rasa Core a
can learn from these how to handle the different situations: a
 a
 a
.. code-block:: story a
 a
   # story1 a
   ... a
   * inform{"people": "3"} a
     - action_book_table a
   ... a
   # story2 a
   * inform{"people": "9"} a
     - action_explain_table_limit a
 a