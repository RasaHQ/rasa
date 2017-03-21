.. _section_pipeline:

Processing Pipeline
===================
The process of incoming messages is split into different components. These components are executed one after another
in a so called processing pipeline. There are components for entity extraction, for intent classification,
pre-processing and there will be many more in the future.

Each component processes the input and creates an output. The ouput can be used by any component that comes after
this component in the pipeline. There are components which only produce information that is used by other components
in the pipeline and there are other components that produce ``Output`` attributes which will be returned after
the processing has finished. E.g. for the sentence ``"I am looking for Chinese food"`` the output

.. code-block:: json

    {
        "text": "I am looking for Chinese food",
        "entities": [
            {"start": 8, "end": 15, "value": "chinese", "entity": "cuisine"}
        ],
        "intent": {"confidence": 0.6485910906220309, "name": "restaurant_search"},
        "intent_ranking": [
            {"confidence": 0.6485910906220309, "name": "restaurant_search"},
            {"confidence": 0.1416153159565678, "name": "affirm"}
        ]
    }

is created as a combination of the results of the different components in the pre-configured pipeline ``spacy_sklearn``.
For example, the ``entities`` attribute is created by the ``ner_spacy`` component.

Pre-configured Pipelines
------------------------
To ease the burden of coming up with your own processing pipelines, we provide a couple of ready to use templates
which can be used by settings the ``pipeline`` configuration value to the name of the template you want to use.
Here is a list of the existing templates:

+---------------+----------------------------------------------------------------------------------------------------------------------------+
| template name | corresponding pipeline                                                                                                     |
+===============+============================================================================================================================+
| spacy_sklearn | ``["init_spacy", "ner_spacy", "ner_synonyms", "intent_featurizer_spacy", "intent_classifier_sklearn"]``                    |
+---------------+----------------------------------------------------------------------------------------------------------------------------+
| mitie         | ``["init_mitie", "tokenizer_mitie", "ner_mitie", "ner_synonyms", "intent_classifier_mitie"]``                              |
+---------------+----------------------------------------------------------------------------------------------------------------------------+
| mitie_sklearn | ``["init_mitie", "tokenizer_mitie", "ner_mitie", "ner_synonyms", "intent_featurizer_mitie", "intent_classifier_sklearn"]`` |
+---------------+----------------------------------------------------------------------------------------------------------------------------+
| keyword       | ``["intent_classifier_keyword"]``                                                                                          |
+---------------+----------------------------------------------------------------------------------------------------------------------------+

Creating your own pipelines is possible by directly passing the names of the components to rasa NLU in the ``pipeline``
configuration variable, e.g. ``"pipeline": ["init_spacy", "ner_spacy", "ner_synonyms"]``. This creates a pipeline
that only does entity recognition, but no intent classification. Hence, the output will not contain any useful intents.

Build-in Components
-------------------

Short explanation of every components and it's attributes. If you are looking for more details, you should have
a look at the corresponding source code for the component. ``Output`` describes, what each component adds to the final
output result of processing a message. If no output is present, the component is most likely a preprocessor for another
component.

init_mitie
~~~~~~~~~~

:Short: MITIE initializer
:Outputs: nothing
:Description:
    Initializes mitie structures. Every mitie component relies on this, hence this should be put at the beginning
    of every pipeline that uses any mitie components.

init_spacy
~~~~~~~~~~

:Short: spacy language initializer
:Outputs: nothing
:Description:
    Initializes spacy structures. Every spacy component relies on this, hence this should be put at the beginning
    of every pipeline that uses any spacy components.

intent_featurizer_mitie
~~~~~~~~~~~~~~~~~~~~~~~

:Short: MITIE intent featurizer
:Outputs: nothing, used used as an input to intent classifiers that need intent features (e.g. ``intent_classifier_sklearn``)
:Description:
    Creates feature for intent classification using the MITIE featurizer.

    .. note::

        NOT used by the ``intent_classifier_mitie`` component. Currently, only ``intent_classifier_sklearn`` is able
        to use precomputed features.


intent_featurizer_spacy
~~~~~~~~~~~~~~~~~~~~~~~

:Short: spacy intent featurizer
:Outputs: nothing, used used as an input to intent classifiers that need intent features (e.g. ``intent_classifier_sklearn``)
:Description:
    Creates feature for intent classification using the spacy featurizer.

intent_featurizer_ngrams
~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Appends char-ngram features to feature vector
:Outputs: nothing, appends its features to an existing feature vector generated by another intent featurizer
:Description:
    This featurizer appends character ngram features to a feature vector. During training the component looks for the
    most common character sequences (e.g. ``app`` or ``ing``). The added features represent a boolean flag if the
    character sequence is present in the word sequence or not.

    .. note:: There needs to be another intent featurizer previous to this one in the pipeline!


intent_classifier_keyword
~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Simple keyword matching intent classifier.
:Outputs: ``intent``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.98343}
        }

:Description:
    This classifier is mostly used as a placeholder. It is able to recognize `hello` and
    `goodbye` intents by searching for these keywords in the passed messages.

intent_classifier_mitie
~~~~~~~~~~~~~~~~~~~~~~~

:Short: MITIE intent classifier (using a `text categorizer <https://github.com/mit-nlp/MITIE/blob/master/examples/python/text_categorizer_pure_model.py>`_)
:Outputs: ``intent``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.98343}
        }

:Description:
    This classifier uses MITIE to perform intent classification. The underlying classifier
    is using a multi class linear SVM with a sparse linear kernel (see `mitie trainer code <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/text_categorizer_trainer.cpp#L222>`_).

intent_classifier_sklearn
~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: sklearn intent classifier
:Outputs: ``intent`` and ``intent_ranking``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.78343},
            "intent_ranking": [
                {
                    "confidence": 0.1485910906220309,
                    "name": "goodbye"
                },
                {
                    "confidence": 0.08161531595656784,
                    "name": "restaurant_search"
                }
            ]
        }

:Description:
    The sklearn intent classifier trains a linear SVM which gets optimized using a grid search. In addition
    to other classifiers it also provides rankings of the labels that did not "win". The spacy intent classifier
    needs to be preceded by a featurizer in the pipeline. This featurizer creates the features used for the classification.

tokenizer_whitespace
~~~~~~~~~~~~~~~~~~~~

:Short: Tokenizer using whitespaces as a separator
:Outputs: nothing
:Description:
    Creates a token for every whitespace separated character sequence. Can be used to define tokesn for the MITIE entity
    extractor.

tokenizer_mitie
~~~~~~~~~~~~~~~

:Short: Tokenizer using MITIE
:Outputs: nothing
:Description:
        Creates tokens using the MITIE tokenizer. Can be used to define tokens for the MITIE entity extractor.

tokenizer_spacy
~~~~~~~~~~~~~~~

:Short: Tokenizer using spacy
:Outputs: nothing
:Description:
        Creates tokens using the spacy tokenizer. Can be used to define tokens for the MITIE entity extractor.


ner_mitie
~~~~~~~~~

:Short: MITIE entity extraction (using a `mitie ner trainer <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/ner_trainer.cpp>`_)
:Outputs: ``entities``
:Output-Example:

    .. code-block:: json

        {
            "entities": [{"value": "New York City", "start": 20, "end": 33, "entity": "city"}]
        }

:Description:
    This uses the MITIE entitiy extraction to find entities in a message. The underlying classifier
    is using a multi class linear SVM with a sparse linear kernel and custom features.

ner_spacy
~~~~~~~~~

:Short: spacy entity extraction
:Outputs: ``entities``
:Output-Example:

    .. code-block:: json

        {
            "entities": [{"value": "New York City", "start": 20, "end": 33, "entity": "city"}]
        }

:Description:
    Using spacy this component predicts the entities of a message. spacy uses a statistical BILUO transition model.
    The entity extractor expects around 5000 training examples per entity to perform good.

ner_synonyms
~~~~~~~~~~~~

:Short: Maps synonymous entity values to the same value.
:Outputs: modifies existing output of a previous entity extraction component

:Description:
    If the training data contains defined synonyms (by using the ``value`` attribute on the entity examples).
    this component will make sure that detected entity values will be mapped to the same value. For example,
    if your training data contains the following examples:

    .. code-block:: json

        [{
          "text": "I moved to New York City",
          "intent": "inform_relocation",
          "entities": [{"value": "nyc", "start": 11, "end": 24, "entity": "city"}]
        },
        {
          "text": "I got a new flat in NYC.",
          "intent": "inform_relocation",
          "entities": [{"value": "nyc", "start": 20, "end": 23, "entity": "city"}]
        }]

    this component will allow you to map the entities ``New York City`` and ``NYC`` to ``nyc``. The entitiy
    extraction will return ``nyc`` even though the message contains ``NYC``.

ner_synonyms
~~~~~~~~~~~~

:Short: MITIE intent classifier (using a `text categorizer <https://github.com/mit-nlp/MITIE/blob/master/examples/python/text_categorizer_pure_model.py>`_)
:Outputs: ``intent``
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.98343}
        }

:Description:
    This classifier uses MITIE to perform intent classification. The underlying classifier
    is using a multi class linear SV; with a sparse linear kernel (see `mitie trainer code <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/text_categorizer_trainer.cpp#L222>`_).


Creating new Components
-----------------------
Currently you need to rely on the components that are shipped with rasa NLU, but we plan to add the possibility to
create your own components in your code. Nevertheless, we are looking forward to your contribution of a new component
(e.g. a component to do sentiment analysis). A glimpse into the code of ``rasa_nlu.components.Component`` will reveal
which functions need to be implemented to create a new component.
