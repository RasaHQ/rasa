:desc: Evaluate and validate your machine learning models for open source
       library Rasa Core to improve the dialogue management of your contextual
       AI Assistant. 

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

    $ rasa test core --core models/dialogue \
      --stories test_stories.md -o results


This will print the failed stories to ``results/failed_stories.md``.
We count any story as `failed` if at least one of the actions
was predicted incorrectly.

In addition, this will save a confusion matrix to a file called
``results/story_confmat.pdf``. The confusion matrix shows, for each action in
your domain, how often that action was predicted, and how often an
incorrect action was predicted instead.

The full list of options for the script is:

.. program-output:: rasa test core --help

.. _end_to_end_evaluation:

End-to-end evaluation of Rasa NLU and Core
------------------------------------------

Say your bot uses a dialogue model in combination with a Rasa NLU model to
parse intent messages, and you would like to evaluate how the two models
perform together on whole dialogues.
The evaluate script lets you evaluate dialogues end-to-end, combining
Rasa NLU intent predictions with Rasa Core action predictions.
You can activate this feature with the ``--e2e`` option in the
``rasa test core`` module.

The story format used for end-to-end evaluation is slightly different to
the standard Rasa Core stories, as you'll have to include the user
messages in natural language instead of just their intent. The format for the
user messages is ``* <intent>:<Rasa NLU example>``. The NLU part follows the
`markdown syntax for Rasa NLU training data
<https://rasa.com/docs/nlu/dataformat/#markdown-format>`_.

Here's an example of what an end-to-end story file may look like:

.. code-block:: story

  ## end-to-end story 1
  * greet: hello
     - utter_ask_howcanhelp
  * inform: show me [chinese](cuisine) restaurants
     - utter_ask_location
  * inform: in [Paris](location)
     - utter_ask_price

  ## end-to-end story 2
  ...


If you've saved these stories under ``e2e_stories.md``,
the full end-to-end evaluation command is this:

.. code-block:: bash

  $ rasa test core -m models --stories e2e_stories.md --e2e

.. note::

  Make sure your model file in ``models`` contains both models, ``core``
  and ``nlu``. If it does not contain a ``nlu`` model, Rasa Core will load
  the default ``RegexInterpreter``.


Comparing Policies
------------------

To choose a specific policy, or to choose hyperparameters for a
specific policy, you want to measure how well Rasa Core will `generalise`
to conversations which it hasn't seen before. Especially in the beginning
of a project, you do not have a lot of real conversations to use to train
your bot, so you don't just want to throw some away to use as a test set.

Rasa Core has some scripts to help you choose and fine-tune your policy.
Once you are happy with it, you can then train your final policy on your
full data set. To do this, you first have to train models for your different
policies. Create two (or more) policy config files of the policies you want to
compare (containing only one policy each), and then use the ``compare`` mode of
the train script to train your models:

.. code-block:: bash

  $ rasa train core -c policy_config1.yml policy_config2.yml \
    -d domain.yml -s stories_folder -o comparison_models --runs 3 --percentages \
    0 5 25 50 70 90 95

For each policy configuration provided, Rasa Core will be trained multiple times
with 0, 5, 25, 50, 70 and 95% of your training stories excluded from the training
data. This is done for multiple runs, to ensure consistent results.

Once this script has finished, you can now use the evaluate script in compare
mode to evaluate the models you just trained:

.. code-block:: bash

  $ rasa test core --stories stories_folder \
    --core comparison_models \
    -o comparison_results

This will evaluate each of the models on the training set, and plot some graphs
to show you which policy is best.  By evaluating on the full set of stories, you
can measure how well Rasa Core is predicting the held-out stories.

If you're not sure which policies to compare, we'd recommend trying out the
``EmbeddingPolicy`` and the ``KerasPolicy`` to see which one works better for
you.

.. note::
    This training process can take a long time, so we'd suggest letting it run
    somewhere in the background where it can't be interrupted


Evaluating stories over http
----------------------------

Rasa Core's server lets you to retrieve evaluations for the currently
loaded model. Say your Rasa Core server is running locally on port 5005,
and your story evaluation file is saved at ``eval_stories.md``. The command
to post stories to the server for evaluation is this:

.. code-block:: bash

  $ curl --data-binary @eval_stories.md "localhost:5005/evaluate" | python -m json.tool

If you would like to evaluate end-to-end stories
(:ref:`docs <end_to_end_evaluation>`),
you may do so by adding the ``e2e=true`` query parameter:

.. code-block:: bash

  $ curl --data-binary @eval_stories.md "localhost:5005/evaluate?e2e=true" | python -m json.tool

.. include:: feedback.inc
