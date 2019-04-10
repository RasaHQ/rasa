:desc: Information about changes between major versions of chatbot framework
       Rasa Core and how you can migrate from one version to another.

.. _migration:

Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

.. _migration-to-0-15-0:

0.14.x to 0.15.0

General
~~~~~~~

- The scripts in ``rasa.core`` and ``rasa.nlu`` can no longer be executed. To train, test, run, ... a rasa nlu or core
  model, you should now use the command line interface ``rasa``. The functionality is the same as before. If you run
  one of the old scripts in ``rasa.core`` or ``rasa.nlu`` an error is thrown that also points you to the command you
  should use instead.
  Mapping of old scripts to new commands:
  ``rasa.core.run`` -> ``rasa shell``
  ``rasa.core.server`` -> ``rasa run core``
  ``rasa.core.test`` -> ``rasa test core``
  ``rasa.core.train`` -> ``rasa train core``
  ``rasa.core.visualize`` -> ``rasa show``
  ``rasa.nlu.convert`` -> ``rasa data``
  ``rasa.nlu.evaluate`` -> ``rasa test nlu``
  ``rasa.nlu.run`` -> ``rasa shell``
  ``rasa.nlu.server`` -> ``rasa run nlu``
  ``rasa.nlu.test`` -> ``rasa test nlu``
  ``rasa.nlu.train`` -> ``rasa train nlu``