.. _slots:

Using Slots
===========

**Slots are your bot's memory.** They act as a key-value store
which can be used to store information the user provided (e.g their home city)
as well as information gathered about the outside world (e.g. the result of a 
database query).

Most of the time, you want slots to influence how the dialogue progresses. 
There are different slot types for different behaviors. 

For example, if your user has provided their home city, you might have a ``text`` slot
called ``home_city``. If the user asks for the weather, and you *don't* know their home 
city, you will have to ask them for it. A ``text`` slot only tells Rasa Core whether
the slot has a value. The specific value of a ``text`` slot
(e.g. Bangalore or New York or Hong Kong) doesn't make any difference.

If the value itself is important, use a ``categorical`` slot. There are
also ``boolean``, ``float``, and ``list`` slots. 
If you just want to store some data, but don't want it to affect the flow
of the conversation, use an ``unfeaturized`` slot. 


How Rasa Uses Slots
-------------------

The :class:`rasa_core.policies.Policy` doesn't have access to the value of your slots.
It receives a ``featurized`` representation. 
As mentioned above, for a ``text`` slot the value is irrelevant. 
The policy just sees a ``1`` or ``0`` depending on whether it is set. 

**You should choose your slot types carefully!**

How Slots Get Set
-----------------

There are two ways that slots may be set:

Slots Set from NLU
~~~~~~~~~~~~~~~~~~

If your NLU model picks up an entity, and your domain contains a slot with the same name, 
the slot will be set automatically. For example:
       
.. code-block:: markdown
   # story_01
   * greet{"name": "Ali"}
     - slot{"name": "Ali"}
     - utter_greet

In this case, you don't have to include the ``- slot{}`` part in the story, because 
it is automatically picked up.


Slots Set by Actions
~~~~~~~~~~~~~~~~~~~~

The second option is to set slots by returning events in :ref:`custom_actions`.
In this case, your stories need to include the slots.
For example, you have a custom action to fetch a user's profile, and 
you have a ``categorical`` slot called ``account_type``:

.. code_block:: yaml

         slots:
            account_type:
               type: categorical
               values:
               - premium
               - basic

.. code-block:: markdown
   # story_02
   * greet
     - action_fetch_profile
     - slot{"account_type" : "premium"}
     - utter_welcome_premium


Slot Types
----------

Here are all of the predefined slot classes and what they're useful for:

.. option:: text

   :Use For: User preferences where you only care whether or not they've
             been specified.
   :Example:
      .. sourcecode:: yaml

         slots:
            cuisine:
               type: text
   :Description:
       :class:`rasa_core.slots.Slot`
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
            risk_level:
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
