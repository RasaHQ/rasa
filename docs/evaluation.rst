:desc: How to evaluate a Rasa Core model

.. _evaluation:

Evaluating and Testing
======================

Evaluating a Trained Model
--------------------------

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


Comparing Policies
------------------

To choose a specific policy, or to choose hyperparameters for a specific policy, you want
to measure how well Rasa Core will `generalise` to conversations which it hasn't seen before.
Especially in the beginning of a project, you do not have a lot of real conversations to use to train
your bot, so you don't just want to throw some away to use as a test set. 

Rasa Core has some scripts to help you choose and fine-tune your policy. Once you are happy
with it, you can then train your final policy on your full data set.
To do this, split your training data into multiple files in a single directory.
You can then use the ``train_paper`` script to train multiple policies on the same data.
You can choose one of the files to be partially excluded. This means that Rasa Core
will be trained multiple times, with 0, 5, 25, 50, 70, 90, 95, and 100% of the stories
in that file removed from the training data. By evaluating on the full set of stories, you
can measure how well Rasa Core is predicting the held-out stories. 


The full list of options for the script is:

.. program-output:: python -m rasa_core.train_paper -h


The full list of options for the evaluation script is:

.. program-output:: python -m rasa_core.evaluate_paper -h

