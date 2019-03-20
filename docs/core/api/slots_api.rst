:desc: Define different slots variables to store contextual information that can
       personalise user experience using machine learning based dialogue
       management. 

.. _slot_types:

Slot Types
==========

Text Slot
---------

.. option:: text

   :Use For: User preferences where you only care whether or not they've
             been specified.
   :Example:
      .. sourcecode:: yaml

         slots:
            cuisine:
               type: text
   :Description:
       :class:`rasa.core.slots.Slot`
       Results in the feature of the slot being set to ``1`` if any value is set.
       Otherwise the feature will be set to ``0`` (no value is set).

Boolean Slot
------------

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
----------------

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
----------

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
---------

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

Unfeaturized Slot
-----------------

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



The Slot base class
-------------------

.. autoclass:: rasa.core.slots.Slot

   .. automethod:: feature_dimensionality

   .. automethod:: as_feature


.. include:: ../feedback.inc
