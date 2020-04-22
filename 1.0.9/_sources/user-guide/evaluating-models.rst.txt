:desc: Evaluate and validate your machine learning models for open source
       library Rasa Core to improve the dialogue management of your contextual
       AI Assistant.

.. _evaluating-models:

Evaluating Models
=================

.. contents::
   :local:

.. note::
   If you are looking to tune the hyperparameters of your NLU model,
   check out this `tutorial <https://blog.rasa.com/rasa-nlu-in-depth-part-3-hyperparameters/>`_.


.. _nlu-evaluation:

Evaluating an NLU Model
-----------------------

A standard technique in machine learning is to keep some data separate as a *test set*.
You can :ref:`split your NLU training data <train-test-split>`
into train and test sets using:

.. code-block:: bash

   rasa data split nlu


If you've done this, you can see how well your NLU model predicts the test cases using this command:

.. code-block:: bash

   rasa test nlu -u test_set.md --model models/nlu-20180323-145833.tar.gz


If you don't want to create a separate test set, you can
still estimate how well your model generalises using cross-validation.
To do this, add the flag ``--cross-validation``:

.. code-block:: bash

   rasa test nlu -u data/nlu.md --config config.yml --cross-validation

The full list of options for the script is:

.. program-output:: rasa test nlu --help



Intent Classification
^^^^^^^^^^^^^^^^^^^^^

The evaluation script will produce a report, confusion matrix,
and confidence histogram for your model.

The report logs precision, recall and f1 measure for
each intent and entity, as well as providing an overall average.
You can save these reports as JSON files using the ``--report`` argument.

The confusion matrix shows you which
intents are mistaken for others; any samples which have been
incorrectly predicted are logged and saved to a file
called ``errors.json`` for easier debugging.

The histogram that the script produces allows you to visualise the
confidence distribution for all predictions,
with the volume of correct and incorrect predictions being displayed by
blue and red bars respectively.
Improving the quality of your training data will move the blue
histogram bars to the right and the red histogram bars
to the left of the plot.


.. note::
    A confusion matrix will **only** be created if you are evaluating a model on a test set.
    In cross-validation mode, the confusion matrix will not be generated.

.. warning::
    If any of your entities are incorrectly annotated, your evaluation may fail. One common problem
    is that an entity cannot stop or start inside a token.
    For example, if you have an example for a ``name`` entity
    like ``[Brian](name)'s house``, this is only valid if your tokenizer splits ``Brian's`` into
    multiple tokens. A whitespace tokenizer would not work in this case.


Entity Extraction
^^^^^^^^^^^^^^^^^

The ``CRFEntityExtractor`` is the only entity extractor which you train using your own data,
and so is the only one that will be evaluated. If you use the spaCy or duckling
pre-trained entity extractors, Rasa NLU will not include these in the evaluation.

Rasa NLU will report recall, precision, and f1 measure for each entity type that
``CRFEntityExtractor`` is trained to recognize.


Entity Scoring
^^^^^^^^^^^^^^

To evaluate entity extraction we apply a simple tag-based approach. We don't consider BILOU tags, but only the
entity type tags on a per token basis. For location entity like "near Alexanderplatz" we
expect the labels ``LOC LOC`` instead of the BILOU-based ``B-LOC L-LOC``. Our approach is more lenient
when it comes to evaluation, as it rewards partial extraction and does not punish the splitting of entities.
For example, given the aforementioned entity "near Alexanderplatz" and a system that extracts
"Alexanderplatz", our approach rewards the extraction of "Alexanderplatz" and punishes the missed out word "near".
The BILOU-based approach, however, would label this as a complete failure since it expects Alexanderplatz
to be labeled as a last token in an entity (``L-LOC``) instead of a single token entity (``U-LOC``). Note also that
a split extraction of "near" and "Alexanderplatz" would get full scores on our approach and zero on the
BILOU-based one.

Here's a comparison between the two scoring mechanisms for the phrase "near Alexanderplatz tonight":

==================================================  ========================  ===========================
extracted                                           Simple tags (score)       BILOU tags (score)
==================================================  ========================  ===========================
[near Alexanderplatz](loc) [tonight](time)          loc loc time (3)          B-loc L-loc U-time (3)
[near](loc) [Alexanderplatz](loc) [tonight](time)   loc loc time (3)          U-loc U-loc U-time (1)
near [Alexanderplatz](loc) [tonight](time)          O   loc time (2)          O     U-loc U-time (1)
[near](loc) Alexanderplatz [tonight](time)          loc O   time (2)          U-loc O     U-time (1)
[near Alexanderplatz tonight](loc)                  loc loc loc  (2)          B-loc I-loc L-loc  (1)
==================================================  ========================  ===========================


.. _core-evaluation:

Evaluating a Core Model
-----------------------

You can evaluate your trained model on a set of test stories
by using the evaluate script:

.. code-block:: bash

    rasa test core --stories test_stories.md --out results


This will print the failed stories to ``results/failed_stories.md``.
We count any story as `failed` if at least one of the actions
was predicted incorrectly.

In addition, this will save a confusion matrix to a file called
``results/story_confmat.pdf``. For each action in your domain, the confusion
matrix shows how often the action was correctly predicted and how often an
incorrect action was predicted instead.

The full list of options for the script is:

.. program-output:: rasa test core --help


Comparing Policies
------------------

To choose a specific policy configuration, or to choose hyperparameters for a
specific policy, you want to measure how well Rasa Core will `generalise`
to conversations which it hasn't seen before. Especially in the beginning
of a project, you do not have a lot of real conversations to use to train
your bot, so you don't just want to throw some away to use as a test set.

Rasa Core has some scripts to help you choose and fine-tune your policy configuration.
Once you are happy with it, you can then train your final configuration on your
full data set. To do this, you first have to train models for your different
policies. Create two (or more) config files including the policies you want to
compare (containing only one policy each), and then use the ``compare`` mode of
the train script to train your models:

.. code-block:: bash

  $ rasa train core -c config_1.yml config_2.yml \
    -d domain.yml -s stories_folder --out comparison_models --runs 3 \
    --percentages 0 5 25 50 70 95

For each policy configuration provided, Rasa Core will be trained multiple times
with 0, 5, 25, 50, 70 and 95% of your training stories excluded from the training
data. This is done for multiple runs to ensure consistent results.

Once this script has finished, you can use the evaluate script in compare
mode to evaluate the models you just trained:

.. code-block:: bash

  $ rasa test core -m comparison_models/<model-1>.tar.gz comparison_models/<model-2>.tar.gz \
    --stories stories_folder --out comparison_results

This will evaluate each of the models on the training set and plot some graphs
to show you which policy performs best.  By evaluating on the full set of stories, you
can measure how well Rasa Core is predicting the held-out stories.

If you're not sure which policies to compare, we'd recommend trying out the
``EmbeddingPolicy`` and the ``KerasPolicy`` to see which one works better for
you.

.. note::
    This training process can take a long time, so we'd suggest letting it run
    somewhere in the background where it can't be interrupted.


.. _end_to_end_evaluation:

End-to-End Evaluation
---------------------

Rasa lets you evaluate dialogues end-to-end, running through
test conversations and making sure that both NLU and Core make correct predictions.

To do this, you need some stories in the end-to-end format,
which includes both the NLU output and the original text.
Here is an example:

.. code-block:: story

  ## end-to-end story 1
  * greet: hello
     - utter_ask_howcanhelp
  * inform: show me [chinese](cuisine) restaurants
     - utter_ask_location
  * inform: in [Paris](location)
     - utter_ask_price


If you've saved end-to-end stories as a file called ``e2e_stories.md``,
you can evaluate your model against them by running:

.. code-block:: bash

  $ rasa test --stories e2e_stories.md --e2e

.. note::

  Make sure your model file in ``models`` is a combined ``core``
  and ``nlu`` model. If it does not contain an NLU model, Core will use
  the default ``RegexInterpreter``.
