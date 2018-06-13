.. _evaluation:

Evaluating Models
=================

You can evaluate your trained model on a set of test stories by using the evaluate script:

.. code-block:: bash

    python -m rasa_core.evaluate -d models/dialogue -s test_stories.md -o matrix.pdf --failed failed_stories.md

This will save a confusion matrix to matrix.pdf and a list of failed stories to failed_stories.md


Dialogue Generalisation
-----------------------

TODO
