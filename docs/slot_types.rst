.. _slot_types:

Slot Types Reference
====================

These are all of the predefined slot classes and what they're useful for.




``type: boolean``
~~~~~~~~~~~~~~~
:Use For: True or False
:params: None
:Description:
    Checks if slot is set and if True

``type: categorical``
~~~~~~~~~~~~~~~~~~~
:Use For: Slots which can take one of N values
:params: ``values``
:Description:
   Creates a one-hot encoding describing which of the ``values`` matched.


``type: data``
~~~~~~~~~~~~
:Use For:  Base class for creating own slots
:Description: 
   User has to subclass this and define the ``as_feature`` method containing any custom logic.
   


``type: float``
~~~~~~~~~~~~~

:Use For: Continuous values
:params: ``max_value``, ``min_value``
:Description:
    Checks if float is within the range of min and max values.


``type: list``
~~~~~~~~~~~~
:Use For: Lists of values
:Description:
    None


``type: text``
~~~~~~~~~~~~
:Use For: User preferences where you only care whether or not they've been specified.
:params: None
:Description:
    Checks if float is within the range of min and max values.


``type: unfeaturized``
~~~~~~~~~~~~~~~~~~~~
:Use For: Data you want to store which shouldn't influence the dialogue flow
:Description:
    




