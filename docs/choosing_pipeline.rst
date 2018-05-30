.. _choosing_pipeline:

Choosing A Pipeline
===================

The two most important pipelines are ``tensorflow_embedding`` and ``spacy_sklearn``.


Tensorflow Embedding or Spacy?

If you want to split intents into multiple labels, e.g. for predicting multiple intents or for modeling hierarchical intent structure, use these flags:

    - ``intent_tokenization_flag`` if ``true`` the algorithm will split the intent labels into tokens and use bag-of-words representations for them;
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
useful intents.

