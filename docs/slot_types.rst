.. _slot_types:

Slot Types Reference
====================

Slots influence the prediction of the next action the bot should run. For the
prediction, the slots value is not used directly, but rather it is featurized.
E.g. for a slot of type ``text``, the value is irrelevant, for the featurization
the only thing that matters is if a text is set or not.

The choice of slot should be done with care. If a slots value should influence
the dialogue flow (e.g. the users age influences which question follows next)
you should choose a slot where the value influences the dialogue model.

These are all of the predefined slot classes and what they're useful for.


``type: boolean``
~~~~~~~~~~~~~~~~~
:Use For: True or False
:Description:
    Checks if slot is set and if True


``type: categorical``
~~~~~~~~~~~~~~~~~~~~~
:Use For: Slots which can take one of N values
:params: ``values``
:Description:
   Creates a one-hot encoding describing which of the ``values`` matched.


``type: data``
~~~~~~~~~~~~~~
:Use For:  Base class for creating own slots
:Description: 
   User has to subclass this and define the ``as_feature`` method containing
   any custom logic.
   

``type: float``
~~~~~~~~~~~~~~~

:Use For: Continuous values
:params: ``max_value``, ``min_value``
:Description:
    Checks if float is within the range of min and max values.


``type: list``
~~~~~~~~~~~~~~
:Use For: Lists of values
:Description:
    The feature of this slot is set to ``1`` if a value with a list is set,
    where the list is not empty. If no value is set, or the empty list is the
    set value, the feature will be ``0``.



``type: text``
~~~~~~~~~~~~~~
:Use For: User preferences where you only care whether or not they've
          been specified.
:Description:
    Results in the feature of the slot being set to ``1`` if any value is set.
    Otherwise the feature will be set to ``0`` (no value is set).


``type: unfeaturized``
~~~~~~~~~~~~~~~~~~~~~~
:Use For: Data you want to store which shouldn't influence the dialogue flow
:Description:
    There will not be any featurization of this slot, hence its value does
    not influence the dialogue flow and is ignored when predicting the next
    action the bot should run.
    




