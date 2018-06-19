.. _evaluation:

Evaluating and Testing
======================

You can evaluate your trained model on a set of test stories by using the evaluate script:

.. code-block:: bash

    python -m rasa_core.evaluate -d models/dialogue -s test_stories.md -o matrix.pdf --failed failed_stories.md


This will print the failed stories to ``failed_stories.md``. 
We count any story as `failed` if at least one of the actions was predicted incorrectly.

In addition, this will save a confusion matrix to a file called ``matrix.pdf``.
The confusion matrix shows, for each action in your domain, how often that action
was predicted, and how often an incorrect action was predicted instead.



The full list of options for the script is:

.. program-output:: python -m rasa_core.evaluate -h
