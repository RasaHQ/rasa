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