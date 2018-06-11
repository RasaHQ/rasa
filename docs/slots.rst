.. _slots:

Using Slots
===========

Slots are your bots memory. 

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
