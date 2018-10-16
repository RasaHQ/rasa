:desc: How to evaluate a Rasa Core model

.. _evaluation:

Evaluating and Testing
======================

.. note::

  If you're looking to evaluate both Rasa NLU and Rasa Core predictions
  combined, take a look at the section on
  :ref:`end-to-end evaluation <end_to_end_evaluation>`.

Evaluating a Trained Model
--------------------------

You can evaluate your trained model on a set of test stories
by using the evaluate script:

.. code-block:: bash

    python -m rasa_core.evaluate -d models/dialogue \
      -s test_stories.md -o matrix.pdf --failed failed_stories.md


This will print the failed stories to ``failed_stories.md``.
We count any story as `failed` if at least one of the actions
was predicted incorrectly.

In addition, this will save a confusion matrix to a file called
``matrix.pdf``. The confusion matrix shows, for each action in your
domain, how often that action was predicted, and how often an
incorrect action was predicted instead.

The full list of options for the script is:

.. program-output:: python -m rasa_core.evaluate -h

.. _end_to_end_evaluation:

End-to-end evaluation of Rasa NLU and Core
------------------------------------------

Say your bot uses a dialogue model in combination with a Rasa NLU model to
parse intent messages, and you would like to evaluate how the two models
perform together on whole dialogues.
The evaluate script lets you evaluate dialogues end-to-end, combining
Rasa NLU intent predictions with Rasa Core action predictions.
You can activate this feature with the ``--e2e`` option in the
``rasa_core.evaluate`` module.

The story format used for end-to-end evaluation is slightly different to
the standard Rasa Core stories, as you'll have to include the user
messages in natural language instead of just their intent. The format for the
user messages is ``* <intent>:<Rasa NLU example>``. The NLU part follows the
`markdown syntax for Rasa NLU training data
<https://rasa.com/docs/nlu/dataformat/#markdown-format>`_.

Here's an example of what an end-to-end story may look like:

.. code-block:: story

  ## end-to-end story 1
  * greet:hello
     - utter_ask_howcanhelp
  * inform:show me [chinese](cuisine) restaurants
     - utter_ask_location
  * inform:in [Paris](location)
     - utter_ask_price
  ...

.. note::

  Make sure you specify an NLU model to load using the dialogue model with the
  ``--nlu`` option in ``rasa_core.evaluate``. If you do not specify an NLU
  model, Rasa Core will load the default ``RegexInterpreter``.


Comparing Policies
------------------

To choose a specific policy, or to choose hyperparameters for a
specific policy, you want to measure how well Rasa Core will `generalise`
to conversations which it hasn't seen before. Especially in the beginning
of a project, you do not have a lot of real conversations to use to train
your bot, so you don't just want to throw some away to use as a test set.

Rasa Core has some scripts to help you choose and fine-tune your policy.
Once you are happy with it, you can then train your final policy on your
full data set. To do this, split your training data into multiple files
in a single directory. You can then use the ``train_paper`` script to
train multiple policies on the same data. You can choose one of the
files to be partially excluded. This means that Rasa Core will be
trained multiple times, with 0, 5, 25, 50, 70, 90, 95, and 100% of
the stories in that file removed from the training data. By evaluating
on the full set of stories, you can measure how well Rasa Core is
predicting the held-out stories.


.. include:: feedback.inc


