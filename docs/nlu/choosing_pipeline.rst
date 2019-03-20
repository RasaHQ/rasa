:desc: Set up a pipeline of pre-trained word vectors form GloVe or fastText
       or fit them specifically on your dataset using the tensorflow pipeline
       for open source NLU.

.. _choosing_pipeline:

Choosing a Rasa NLU Pipeline
============================

Choosing an NLU pipeline allows you to customize your model and finetune
it on your dataset.

The Short Answer
----------------

If you have less than 1000 total training examples, and there is a spaCy model for your 
language, use the ``pretrained_embeddings_spacy`` pipeline:

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_spacy.yml
    :language: yaml


If you have 1000 or more labelled utterances,
use the ``supervised_embeddings`` pipeline:

.. code-block:: yaml

    language: "en"

    pipeline: "supervised_embeddings"

It's good practice to define the ``language`` parameter in your configuration, but
for the ``supervised_embeddings`` pipeline this parameter doesn't affect anything.

A Longer Answer
---------------

The two most important pipelines are ``supervised_embeddings`` and ``pretrained_embeddings_spacy``.
The biggest difference between them is that the ``pretrained_embeddings_spacy`` pipeline uses pre-trained
word vectors from either GloVe or fastText. Instead, the supervised embeddings pipeline
doesn't use any pre-trained word vectors, but instead fits these specifically for your dataset.

The advantage of the ``pretrained_embeddings_spacy`` pipeline is that if you have a training example like:
"I want to buy apples", and Rasa is asked to predict the intent for "get pears", your model
already knows that the words "apples" and "pears" are very similar. This is especially useful
if you don't have very much training data.

The advantage of the ``supervised_embeddings`` pipeline is that your word vectors will be customised
for your domain. For example, in general English, the word "balance" is closely related to "symmetry",
but very different to the word "cash". In a banking domain, "balance" and "cash" are closely related
and you'd like your model to capture that. This pipeline doesn't use a language-specific model,
so it will work with any language that you can tokenize (on whitespace or using a custom tokenizer).

You can read more about this topic `here <https://medium.com/rasa-blog/supervised-word-vectors-from-scratch-in-rasa-nlu-6daf794efcd8>`__ .


There are also the ``mitie`` and ``mitie_sklearn`` pipelines, which use MITIE as a source of word vectors.
We do not recommend that you use these; they are likely to be deprecated in a future release.

.. note::

    Intent classification is independent of entity extraction. So sometimes
    NLU will get the intent right but entities wrong, or the other way around.
    You need to provide enough data for both intents and entities.


Multiple Intents
----------------

If you want to split intents into multiple labels,
e.g. for predicting multiple intents or for modeling hierarchical intent structure,
you can only do this with the supervised embeddings pipeline.
To do this, use these flags:

    - ``intent_tokenization_flag`` if ``true`` the algorithm will split the intent labels into tokens and use a bag-of-words representations for them;
    - ``intent_split_symbol`` sets the delimiter string to split the intent labels. Default ``_``

`Here <https://blog.rasa.com/how-to-handle-multiple-intents-per-input-using-rasa-nlu-tensorflow-pipeline/>`__ is a tutorial on how to use multiple intents in Rasa Core and NLU.

Here's an example configuration:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "CountVectorsFeaturizer"
    - name: "EmbeddingIntentClassifier"
      intent_tokenization_flag: true
      intent_split_symbol: "+"



Understanding the Rasa NLU Pipeline
-----------------------------------

In Rasa NLU, incoming messages are processed by a sequence of components.
These components are executed one after another
in a so-called processing pipeline. There are components for entity extraction, for intent classification,
pre-processing, and others. If you want to add your own component, for example to run a spell-check or to
do sentiment analysis, check out :ref:`section_customcomponents`.

Each component processes the input and creates an output. The ouput can be used by any component that comes after
this component in the pipeline. There are components which only produce information that is used by other components
in the pipeline and there are other components that produce ``Output`` attributes which will be returned after
the processing has finished. For example, for the sentence ``"I am looking for Chinese food"`` the output is:

.. code-block:: json

    {
        "text": "I am looking for Chinese food",
        "entities": [
            {"start": 8, "end": 15, "value": "chinese", "entity": "cuisine", "extractor": "CRFEntityExtractor", "confidence": 0.864}
        ],
        "intent": {"confidence": 0.6485910906220309, "name": "restaurant_search"},
        "intent_ranking": [
            {"confidence": 0.6485910906220309, "name": "restaurant_search"},
            {"confidence": 0.1416153159565678, "name": "affirm"}
        ]
    }

This is created as a combination of the results of the different components in the pre-configured pipeline ``pretrained_embeddings_spacy``.
For example, the ``entities`` attribute is created by the ``CRFEntityExtractor`` component.


.. _section_component_lifecycle:

Component Lifecycle
-------------------
Every component can implement several methods from the ``Component``
base class; in a pipeline these different methods
will be called in a specific order. Lets assume, we added the following
pipeline to our config:
``"pipeline": ["Component A", "Component B", "Last Component"]``.
The image shows the call order during the training of this pipeline :

.. image:: _static/images/component_lifecycle.png

Before the first component is created using the ``create`` function, a so
called ``context`` is created (which is nothing more than a python dict).
This context is used to pass information between the components. For example,
one component can calculate feature vectors for the training data, store
that within the context and another component can retrieve these feature
vectors from the context and do intent classification.

Initially the context is filled with all configuration values, the arrows
in the image show the call order and visualize the path of the passed
context. After all components are trained and persisted, the
final context dictionary is used to persist the model's metadata.



Returned Entities Object
------------------------
In the object returned after parsing there are two fields that show information
about how the pipeline impacted the entities returned. The ``extractor`` field
of an entity tells you which entity extractor found this particular entity.
The ``processors`` field contains the name of components that altered this
specific entity.

The use of synonyms can also cause the ``value`` field not match the ``text``
exactly. Instead it will return the trained synonym.

.. code-block:: json

    {
      "text": "show me chinese restaurants",
      "intent": "restaurant_search",
      "entities": [
        {
          "start": 8,
          "end": 15,
          "value": "chinese",
          "entity": "cuisine",
          "extractor": "CRFEntityExtractor",
          "confidence": 0.854,
          "processors": []
        }
      ]
    }

.. note::

    The ``confidence`` will be set by the CRF entity extractor
    (``CRFEntityExtractor`` component). The duckling entity extractor will always return
    ``1``. The ``SpacyEntityExtractor`` extractor does not provide this information and
    returns ``null``.


Pre-configured Pipelines
------------------------

A template is just a shortcut for
a full list of components. For example, these two configurations are equivalent:

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_spacy.yml
    :language: yaml

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "SpacyNLP"
    - name: "SpacyTokenizer"
    - name: "SpacyFeaturizer"
    - name: "RegexFeaturizer"
    - name: "CRFEntityExtractor"
    - name: "EntitySynonymMapper"
    - name: "SklearnIntentClassifier"

Below is a list of all the pre-configured pipeline templates.

.. _section_pretrained_embeddings_spacy_pipeline:

pretrained_embeddings_spacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the ``pretrained_embeddings_spacy`` template:

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_spacy.yml
    :language: yaml

See :ref:`section_languages` for possible values for ``language``. To use
the components and configure them separately:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "SpacyNLP"
    - name: "SpacyTokenizer"
    - name: "SpacyFeaturizer"
    - name: "RegexFeaturizer"
    - name: "CRFEntityExtractor"
    - name: "EntitySynonymMapper"
    - name: "SklearnIntentClassifier"

.. _section_supervised_embeddings_pipeline:

supervised_embeddings
~~~~~~~~~~~~~~~~~~~~~

To use it as a template:

.. code-block:: yaml

    language: "en"

    pipeline: "supervised_embeddings"

The supervised embeddings pipeline supports any language that can be tokenized. The
default is to use a simple whitespace tokenizer:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "WhitespaceTokenizer"
    - name: "CRFEntityExtractor"
    - name: "EntitySynonymMapper"
    - name: "CountVectorsFeaturizer"
    - name: "EmbeddingIntentClassifier"

If you have a custom tokenizer for your language, you can replace the whitespace
tokenizer with something more accurate.

.. _section_mitie_pipeline:

mitie
~~~~~

There is no pipeline template, as you need to configure the location
of MITIE's featurizer. To use the components and configure them separately:

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie.yml
    :language: yaml

mitie_2
~~~~~~~~~~~~~

This pipeline uses MITIE's featurizer and also its multiclass classifier.
Training can be quite slow, so this is not recommended for large datasets.
There is no pipeline template, as you need to configure the location
of MITIE's featurizer. To use the components and configure them separately:

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie_2.yml
    :language: yaml

keyword
~~~~~~~

To use it as a template:

.. code-block:: yaml

    language: "en"

    pipeline: "keyword"

To use the components and configure them separately:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "KeywordIntentClassifier"



Custom pipelines
----------------

You don't have to use a template, you can run a fully custom pipeline
by listing the names of the components you want to use:

.. code-block:: yaml

    pipeline:
    - name: "SpacyNLP"
    - name: "CRFEntityExtractor"
    - name: "EntitySynonymMapper"

This creates a pipeline that only does entity recognition, but no
intent classification. So Rasa NLU will not predict any intents.
You can find the details of each component in :ref:`section_pipeline`.

If you want to use custom components in your pipeline, see :ref:`section_customcomponents`.


.. include:: feedback.inc
