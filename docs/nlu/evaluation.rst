.. _nlu-evaluation:

Evaluating NLU Models
=====================

How is your model performing? Do you have enough data? Are your intents and entities well-designed?

Rasa NLU has an ``test`` mode which helps you answer these questions.
A standard technique in machine learning is to keep some data separate as a *test set*.
If you've done this, you can see how well your model predicts the test cases using this command:

To evaluate your model against test data, run:

.. code-block:: shell

   rasa test nlu


If you don't have a separate test set, you can
still estimate how well your model generalises using cross-validation.
To do this, add the ``--mode crossvalidation`` flag:

.. code-block:: shell

   rasa test nlu --mode crossvalidation


Intent Classification
^^^^^^^^^^^^^^^^^^^^^

The evaluation script will produce a report, confusion matrix
and confidence histogram for your model.

The report logs precision, recall and f1 measure for
each intent and entity, as well as provide an overall average.
You can save these reports as JSON files using the `--report` flag.

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
and so is the only one which will be evaluated. If you use the spaCy or duckling
pre-trained entity extractors, Rasa NLU will not include these in the evaluation.

Rasa NLU will report recall, precision, and f1 measure for each entity type that
``CRFEntityExtractor`` is trained to recognize.


Entity Scoring
^^^^^^^^^^^^^^

To evaluate entity extraction we apply a simple tag-based approach. We don't consider BILOU tags, but only the
entity type tags on a per token basis. For location entity like "near Alexanderplatz" we
expect the labels ``LOC LOC`` instead of the BILOU-based ``B-LOC L-LOC``. Our approach is more lenient
when it comes to evaluation, as it rewards partial extraction and does not punish the splitting of entities.
For example, the given the aforementioned entity "near Alexanderplatz" and a system that extracts
"Alexanderplatz", this reward the extraction of "Alexanderplatz" and punish the missed out word "near".
The BILOU-based approach, however, would label this as a complete failure since it expects Alexanderplatz
to be labeled as a last token in an entity (``L-LOC``) instead of a single token entity (``U-LOC``). Also note,
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


