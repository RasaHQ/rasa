.. _section_evaluation:

Evaluating and Improving Models
===============================

Improving your models from feedback
-----------------------------------

Once you have a version of your bot running, the Rasa NLU server will log 
every request made to the ``/parse`` endpoint to a file. By default
these are saved in the folder ``logs``. 


.. code-block:: javascript

   {  
     "user_input":{  
       "entities":[]   ],
       "intent":{  
         "confidence":0.32584617693743012,
         "name":"restaurant_search"
       },
       "text":"nice thai places",
       "intent_ranking":[ ... ]
     },
     ...
     "model":"default",
     "log_time":1504092543.036279
   }


The things your users say are the best source of training data for refining your models.
Of course your model won't be perfect, so you will have to manually go through
each of these predictions and correct any mistakes before adding them to your training data.
In this case, the entity 'thai' was not picked up as a cuisine. 


Evaluating Models
-----------------

How is your model performing? Do you have enough data? Are your intents and entities well-designed?

Rasa NLU has an ``evaluate`` mode which helps you answer these questions.
A standard technique in machine learning is to keep some data separate as a *test set*.
If you've done this, you can see how well your model predicts the test cases using this command:


.. code-block:: bash

    python -m rasa_nlu.evaluate \
        --data data/examples/rasa/demo-rasa.json \
        --model projects/default/model_20180323-145833

Where the ``--data`` argument points to your test data, and ``--model`` points to your trained model.


If you don't have a separate test set, you can 
still estimate how well your model generalises using cross-validation. 
To do this, run the evaluation script with the ``--mode crossvalidation`` flag. 


.. code-block:: bash

    python -m rasa_nlu.evaluate \
        --data data/examples/rasa/demo-rasa.json \
        --config sample_configs/config_spacy.yml \
        --mode crossvalidation


You cannot specify a model in this mode because
a new model will be trained on part of the data
for every cross-validation fold.

Example Output
^^^^^^^^^^^^^^




Intent Classification
---------------------
The evaluation script will log precision, recall, and f1 measure for
each intent and once summarized for all.
Furthermore, it creates a confusion matrix for you to see which
intents are mistaken for which others.


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
fit to your training data.


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


Evaluation Parameters
---------------------

There are a number of parameters you can pass to the evaluation script

.. code-block:: bash

    $ python -m rasa_nlu.evaluate --help

Here is a quick overview:

.. program-output:: python -m rasa_nlu.evaluate --help


