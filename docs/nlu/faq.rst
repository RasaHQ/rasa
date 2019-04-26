:desc: Read more about language support, number of training data examples and
       other FAQ's about open source NLP library Rasa NLU.

.. _section_faq:

Frequently Asked Questions
==========================

Which languages does the Rasa NLU support?
------------------------------------------
Rasa NLU can be used to understand any language that can be
tokenized (on whitespace or using a custom tokenizer),
but some backends are restricted to specific languages.

The ``supervised_embeddings`` pipeline can be used for any language because it trains custom word embeddings for your domain using the data you provide in the NLU training examples.

Other backends use pre-trained word vectors and therefore are
restricted to languages which have pre-trained vectors available.

You can read more about the Rasa NLU supported languages in
:ref:`section_languages`.


How many training examples do I need?
-------------------------------------
Unfortunately, the answer is *it depends*.

A good starting point is to have 10 examples for each intent
and build up from there.

If you have intents that are easily confusable, you will need more
training data. Accordingly, as you add more
intents, you also want to add more training examples for each intent.
If you quickly write 20-30 unique expressions for
each intent, you should be good for the beginning.

The same holds true for entities. the number of training examples you
will need depends on how closely related your different entity types
are and how clearly entities are distinguishable from non-entities in
your use case.

To assess your model's performance, use the
:ref:`evaluation script <section_evaluation>`.


.. _section_faq_version:

Which version of Rasa am I running?
-----------------------------------
To find out which Rasa version you are running, you can execute

.. code-block:: bash

   python -c "import rasa; print(rasa.__version__);"

If you are using a virtual environment to run your python code, make sure
you are using the correct python to execute the above code.

Why am I getting an ``UndefinedMetricWarning``?
-----------------------------------------------
The complete warning is:
``UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.``
The warning is a result of a lack of training data. During the training
the dataset will be splitted multiple times, if there are to few training
samples for any of the intents, the splitting might result in splits that
do not contain any examples for this intent.

Hence, the solution is to add more training samples. As this is only a
warning, training will still succeed, but the resulting models predictions
might be weak on the intents where you are lacking training data.  


I have an issue, can you help me?
---------------------------------
We'd love to help you. If you are unsure if your issue is related to your
setup, you should state your problem in the
`Rasa Community Forum <https://forum.rasa.com>`_.
If you found an issue with the framework, please file a report on
`github issues <https://github.com/RasaHQ/rasa_nlu/issues>`_
including all the information needed to reproduce the problem.


Does it run with python 3?
--------------------------
Yes it does, Rasa NLU supports python 3.5, 3.6 and 3.7 (supported for python 2.7
up until version 0.14). If there are any issues with a specific python version,
feel free to create an issue or directly provide a fix.


.. include:: feedback.inc
