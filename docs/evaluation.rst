.. _section_evaluation:

Evaluation
==========

The evaluation script `evaluate.py` allows you to test your models performance for intent classification and entity recognition. You invoke this script supplying test data, model, and config file arguments:

.. code-block:: bash

    python -m rasa_nlu.evaluate -d data/my_test.json -m models/my_model -c my_nlu_config.json 

If you would like to evaluate your pipeline using crossvalidation, you can run the evaluation script with the mode crossvalidation flag. This gives you an estimate of how accurately a predictive model will perform in practice. Note that you cannot specify a model in this mode, as a new model will be trained on part of the data for every crossvalidation loop. An example invocation of your script would be:

.. code-block:: bash

    python -m rasa_nlu.evaluate -d data/examples/rasa/demo-rasa.json -c sample_configs/config_spacy.json --mode crossvalidation

Intent Classification
---------------------
The evaluation script will log precision, recall, and f1 measure for each intent and once summarized for all.
Furthermore, it creates a confusion matrix for you to see which intents are mistaken for which others.

Entity Extraction
-----------------
For each entity extractor, the evaluation script logs its performance per entity type in your training data.
So if you use ``ner_crf`` and ``ner_duckling`` in your pipeline, it will log two evaluation tables
containing recall, precision, and f1 measure for each entity type.

In the case ``ner_duckling`` we actually run the evaluation for each defined
duckling dimension. If you use the ``time`` and ``ordinal`` dimensions, you would
get two evaluation tables: one for ``ner_duckling (Time)`` and one for
``ner_duckling (Ordinal)``.

``ner_synonyms`` does not create an evaluation table, because it only changes the value of the found
entities and does not find entity boundaries itself.

Finally, keep in mind that entity types in your testing data have to match the output
of the extraction components. This is particularly important for ``ner_duckling``, because it is not
fitted to your training data.


Entity Scoring
^^^^^^^^^^^^^^
To evaluate entity extraction we apply a simple tag-based approach. We don't consider BILOU tags, but only the
entity type tags on a per token basis. For location entity like "near Alexanderplatz" we
expect the labels "LOC" "LOC" instead of the BILOU-based "B-LOC" "L-LOC". Our approach is more lenient
when it comes to evaluation, as it rewards partial extraction and does not punish the splitting of entities.
For example, the given the aforementioned entity "near Alexanderplatz" and a system that extracts
"Alexanderplatz", this reward the extraction of "Alexanderplatz" and punish the missed out word "near".
The BILOU-based approach, however, would label this as a complete failure since it expects Alexanderplatz
to be labeled as a last token in an entity (L-LOC) instead of a single token entity (U-LOC). Also note,
a splitted extraction of "near" and "Alexanderplatz" would get full scores on our approach and zero on the
BILOU-based one.

Here's a comparison between both different scoring mechanisms for the phrase "near Alexanderplatz tonight":

==================================================  ========================  ===========================
extracted                                           Simple tags (score)       BILOU tags (score)
==================================================  ========================  ===========================
[near Alexanderplatz](loc) [tonight](time)          loc loc time (3)          B-loc L-loc U-time (3)
[near](loc) [Alexanderplatz](loc) [tonight](time)   loc loc time (3)          U-loc U-loc U-time (1)
near [Alexanderplatz](loc) [tonight](time)          O   loc time (2)          O     U-loc U-time (1)
[near](loc) Alexanderplatz [tonight](time)          loc O   time (2)          U-loc O     U-time (1)
[near Alexanderplatz tonight](loc)                  loc loc loc  (2)          B-loc I-loc L-loc  (1)
==================================================  ========================  ===========================

