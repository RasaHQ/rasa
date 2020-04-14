:desc: Store information the user provided as well as information from database
       queries in slots to influence how the machine learning based dialogue
       continues.

.. _slots:

Slots
=====

**Slots are your bot's memory.** They act as a key-value store
which can be used to store information the user provided (e.g their home city)
as well as information gathered about the outside world (e.g. the result of a
database query).

Most of the time, you want slots to influence how the dialogue progresses.
There are different slot types for different behaviors.

For example, if your user has provided their home city, you might
have a ``text`` slot called ``home_city``. If the user asks for the
weather, and you *don't* know their home city, you will have to ask
them for it. A ``text`` slot only tells Rasa Core whether the slot
has a value. The specific value of a ``text`` slot (e.g. Bangalore
or New York or Hong Kong) doesn't make any difference.

If the value itself is important, use a ``categorical`` or a ``bool`` slot.
There are also ``float``, and ``list`` slots.
If you just want to store some data, but don't want it to affect the flow
of the conversation, use an ``unfeaturized`` slot.


How Rasa Uses Slots
-------------------

The ``Policy`` doesn't have access to the
value of your slots. It receives a featurized representation.
As mentioned above, for a ``text`` slot the value is irrelevant.
The policy just sees a ``1`` or ``0`` depending on whether it is set.

**You should choose your slot types carefully!**

How Slots Get Set
-----------------

You can provide an initial value for a slot in your domain file:

.. code-block:: yaml

    slots:
      name:
        type: text
        initial_value: "human"


There are multiple ways that slots are set during a conversation:

Slots Set from NLU
~~~~~~~~~~~~~~~~~~

If your NLU model picks up an entity, and your domain contains a
slot with the same name, the slot will be set automatically. For example:

.. code-block:: story

   # story_01
   * greet{"name": "Ali"}
     - slot{"name": "Ali"}
     - utter_greet

In this case, you don't have to include the ``- slot{}`` part in the
story, because it is automatically picked up.


Slots Set By Clicking Buttons
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use buttons as a shortcut.
Rasa Core will send messages starting with a ``/`` to the
``RegexInterpreter``, which expects NLU input in the same format
as in story files, e.g. ``/intent{entities}``. For example, if you let
users choose a color by clicking a button, the button payloads might
be ``/choose{"color": "blue"}`` and ``/choose{"color": "red"}``.

You can specify this in your domain file like this:
(see details in :ref:`domains`)

.. code-block:: yaml

  utter_ask_color:
  - text: "what color would you like?"
    buttons:
    - title: "blue"
      payload: '/choose{"color": "blue"}'
    - title: "red"
      payload: '/choose{"color": "red"}'


Slots Set by Actions
~~~~~~~~~~~~~~~~~~~~

The second option is to set slots by returning events in :ref:`custom actions <custom-actions>`.
In this case, your stories need to include the slots.
For example, you have a custom action to fetch a user's profile, and
you have a ``categorical`` slot called ``account_type``.
When the ``fetch_profile`` action is run, it returns a
:class:`rasa.core.events.SlotSet` event:

.. code-block:: yaml

   slots:
      account_type:
         type: categorical
         values:
         - premium
         - basic

.. code-block:: python

   from rasa_sdk.actions import Action
   from rasa_sdk.events import SlotSet
   import requests

   class FetchProfileAction(Action):
       def name(self):
           return "fetch_profile"

       def run(self, dispatcher, tracker, domain):
           url = "http://myprofileurl.com"
           data = requests.get(url).json
           return [SlotSet("account_type", data["account_type"])]


.. code-block:: story

   # story_01
   * greet
     - action_fetch_profile
     - slot{"account_type" : "premium"}
     - utter_welcome_premium

   # story_02
   * greet
     - action_fetch_profile
     - slot{"account_type" : "basic"}
     - utter_welcome_basic


In this case you **do** have to include the ``- slot{}`` part in your stories.
Rasa Core will learn to use this information to decide on the correct action to
take (in this case, ``utter_welcome_premium`` or ``utter_welcome_basic``).

.. note::
   It is **very easy** to forget about slots if you are writing
   stories by hand. We strongly recommend that you build up these
   stories using :ref:`section_interactive_learning_forms` rather than writing them.


.. _slot-classes:

Slot Types
----------

Text Slot
~~~~~~~~~

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

Boolean Slot
~~~~~~~~~~~~

.. option:: bool

  :Use For: True or False
  :Example:
     .. sourcecode:: yaml

        slots:
           is_authenticated:
              type: bool
  :Description:
      Checks if slot is set and if True

Categorical Slot
~~~~~~~~~~~~~~~~

.. option:: categorical

  :Use For: Slots which can take one of N values
  :Example:
     .. sourcecode:: yaml

        slots:
           risk_level:
              type: categorical
              values:
              - low
              - medium
              - high

  :Description:
     Creates a one-hot encoding describing which of the ``values`` matched.

Float Slot
~~~~~~~~~~

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

List Slot
~~~~~~~~~

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

.. _unfeaturized-slot:

Unfeaturized Slot
~~~~~~~~~~~~~~~~~

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

Custom Slot Types
-----------------

Maybe your restaurant booking system can only handle bookings
for up to 6 people. In this case you want the *value* of the
slot to influence the next selected action (and not just whether
it's been specified). You can do this by defining a custom slot class.

In the code below, we define a slot class called ``NumberOfPeopleSlot``.
The featurization defines how the value of this slot gets converted to a vector
to our machine learning model can deal with.
Our slot has three possible "values", which we can represent with
a vector of length ``2``.

+---------------+------------------------------------------+
| ``(0,0)``     | not yet set                              |
+---------------+------------------------------------------+
| ``(1,0)``     | between 1 and 6                          |
+---------------+------------------------------------------+
| ``(0,1)``     | more than 6                              |
+---------------+------------------------------------------+


.. testcode::

   from rasa.core.slots import Slot

   class NumberOfPeopleSlot(Slot):

       def feature_dimensionality(self):
           return 2

       def as_feature(self):
           r = [0.0] * self.feature_dimensionality()
           if self.value:
               if self.value <= 6:
                   r[0] = 1.0
               else:
                   r[1] = 1.0
       return r

Now we also need some training stories, so that Rasa Core
can learn from these how to handle the different situations:


.. code-block:: story

   # story1
   ...
   * inform{"people": "3"}
     - action_book_table
   ...
   # story2
   * inform{"people": "9"}
     - action_explain_table_limit
