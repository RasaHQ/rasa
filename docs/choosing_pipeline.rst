.. _choosing_pipeline:

Choosing a Pipeline
===================


Pre-trained or custom word vectors?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The two most important pipelines are ``tensorflow_embedding`` and ``spacy_sklearn``.
The biggest difference between them is that the ``spacy_sklearn`` pipeline uses pre-trained
word vectors from either GloVe or fastText. Instead, the tensorflow embedding pipeline
doesn't use any pre-trained word vectors, but instead fits these specifically for your dataset.

The advantage of the ``spacy_sklearn`` pipeline is that if you have a training example like:
"I want to buy apples", and Rasa is asked to predict the intent for "I want to buy pears", your model
already knows that the words "apples" and "pears" are very similar. This is especially useful
if you don't have very much training data (< 500 labeled examples). 

The advantage of the ``tensorflow_embedding`` pipeline is that your word vectors will be customised 
for your domain. For example, in general English, the word "balance" is closely related to "symmetry",
but very different to the word "cash". In a banking domain, "balance" and "cash" are closely related
and you'd like your model to capture that.


You can read more about this topic `here <https://medium.com/rasa-blog/supervised-word-vectors-from-scratch-in-rasa-nlu-6daf794efcd8>`_ . 


As a rule of thumb, if there is a spaCy model for your language, 
then the ``spacy_sklearn`` pipeline is a good choice for getting started. 
However once you have more training data (>500 sentences), 
it is highly recommended that you try the ``tensorflow_embedding`` pipeline.

There are also the ``mitie`` and ``mitie_sklearn`` pipelines, which use MITIE as a source of word vectors. 
We do not recommend that you use these; they are likely to be deprecated in a future release.


Multiple Intents
^^^^^^^^^^^^^^^^


If you want to split intents into multiple labels, 
e.g. for predicting multiple intents or for modeling hierarchical intent structure,
you can only do this with the tensorflow pipeline.
To do this, use these flags:

    - ``intent_tokenization_flag`` if ``true`` the algorithm will split the intent labels into tokens and use a bag-of-words representations for them;
    - ``intent_split_symbol`` sets the delimiter string to split the intent labels. Default ``_``


Here's an example configuration:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "intent_featurizer_count_vectors"
    - name: "intent_classifier_tensorflow_embedding"
      intent_tokenization_flag: true
      intent_split_symbol: "_"



Custom pipelines
~~~~~~~~~~~~~~~~

Creating your own pipelines is possible by directly passing the names of the
components to Rasa NLU in the ``pipeline`` configuration variable, e.g.

.. code-block:: yaml

    pipeline:
    - name: "nlp_spacy"
    - name: "ner_crf"
    - name: "ner_synonyms"

This creates a pipeline that only does entity recognition, but no
intent classification. Hence, the output will not contain any
useful intents. You can find the details of each component in :ref:`section_pipeline`.

If you want to use custom components in your pipeline, see :ref:`section_customcomponents`. 
