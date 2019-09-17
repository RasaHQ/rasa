:desc: Set up a pipeline of pre-trained word vectors form GloVe or fastText
       or fit them specifically on your dataset using the TensorFlow pipeline
       for open source NLU.

.. _choosing-a-pipeline:

Choosing a Pipeline
===================

.. edit-link::

Choosing an NLU pipeline allows you to customize your model and finetune
it on your dataset.

.. contents::
   :local:


The Short Answer
----------------

If you have less than 1000 total training examples, and there is a spaCy model for your
language, use the ``pretrained_embeddings_spacy`` pipeline:

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_spacy.yml
    :language: yaml


If you have 1000 or more labelled utterances,
use the ``supervised_embeddings`` pipeline:

.. literalinclude:: ../../sample_configs/config_supervised_embeddings.yml
    :language: yaml


A Longer Answer
---------------

The two most important pipelines are ``supervised_embeddings`` and ``pretrained_embeddings_spacy``.
The biggest difference between them is that the ``pretrained_embeddings_spacy`` pipeline uses pre-trained
word vectors from either GloVe or fastText. The ``supervised_embeddings`` pipeline, on the other hand,
doesn't use any pre-trained word vectors, but instead fits these specifically for your dataset.


pretrained_embeddings_spacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The advantage of the ``pretrained_embeddings_spacy`` pipeline is that if you have a training example like:
"I want to buy apples", and Rasa is asked to predict the intent for "get pears", your model
already knows that the words "apples" and "pears" are very similar. This is especially useful
if you don't have very much training data.

supervised_embeddings
~~~~~~~~~~~~~~~~~~~~~

The advantage of the ``supervised_embeddings`` pipeline is that your word vectors will be customised
for your domain. For example, in general English, the word "balance" is closely related to "symmetry",
but very different to the word "cash". In a banking domain, "balance" and "cash" are closely related
and you'd like your model to capture that. This pipeline doesn't use a language-specific model,
so it will work with any language that you can tokenize (on whitespace or using a custom tokenizer).

You can read more about this topic `here <https://medium.com/rasa-blog/supervised-word-vectors-from-scratch-in-rasa-nlu-6daf794efcd8>`__ .

MITIE
~~~~~

You can also use MITIE as a source of word vectors in your pipeline, see :ref:`section_mitie_pipeline`. The MITIE backend performs well for small datasets, but training can take very long if you have more than a couple of hundred examples.

However, we do not recommend that you use it as mitie support is likely to be deprecated in a future release.

Comparing different pipelines for your data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rasa gives you the tools to compare the performance of both of these pipelines on your data directly,
see :ref:`comparing-nlu-pipelines`.

.. note::

    Intent classification is independent of entity extraction. So sometimes
    NLU will get the intent right but entities wrong, or the other way around.
    You need to provide enough data for both intents and entities.


Class imbalance
---------------

Classification algorithms often do not perform well if there is a large `class imbalance`,
for example if you have a lot of training data for some intents and very little training data for others.
To mitigate this problem, rasa's ``supervised_embeddings`` pipeline uses a ``balanced`` batching strategy.
This algorithm ensures that all classes are represented in every batch, or at least in
as many subsequent batches as possible, still mimicking the fact that some classes are more frequent than others.
Balanced batching is used by default. In order to turn it off and use a classic batching strategy include
``batch_strategy: sequence`` in your config file.

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "CountVectorsFeaturizer"
    - name: "EmbeddingIntentClassifier"
      batch_strategy: sequence


Multiple Intents
----------------

If you want to split intents into multiple labels,
e.g. for predicting multiple intents or for modeling hierarchical intent structure,
you can only do this with the supervised embeddings pipeline.
To do this, use these flags in ``Whitespace Tokenizer``:

    - ``intent_split_symbol``: sets the delimiter string to split the intent labels. Default ``_``

`Here <https://blog.rasa.com/how-to-handle-multiple-intents-per-input-using-rasa-nlu-tensorflow-pipeline/>`__ is a tutorial on how to use multiple intents in Rasa Core and NLU.

Here's an example configuration:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "WhitespaceTokenizer"
      intent_split_symbol: "_"
    - name: "CountVectorsFeaturizer"
    - name: "EmbeddingIntentClassifier"


Understanding the Rasa NLU Pipeline
-----------------------------------

In Rasa NLU, incoming messages are processed by a sequence of components.
These components are executed one after another
in a so-called processing pipeline. There are components for entity extraction, for intent classification, response selection,
pre-processing, and others. If you want to add your own component, for example to run a spell-check or to
do sentiment analysis, check out :ref:`custom-nlu-components`.

Each component processes the input and creates an output. The output can be used by any component that comes after
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
The image shows the call order during the training of this pipeline:

.. image:: /_static/images/component_lifecycle.png

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


The "entity" object explained
-----------------------------
After parsing, the entity is returned as a dictionary.  There are two fields that show information
about how the pipeline impacted the entities returned: the ``extractor`` field
of an entity tells you which entity extractor found this particular entity, and
the ``processors`` field contains the name of components that altered this
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

Below is a list of all the pre-configured pipeline templates with customization information.

.. _section_supervised_embeddings_pipeline:

supervised_embeddings
~~~~~~~~~~~~~~~~~~~~~

To train a Rasa model in your preferred language, define the
``supervised_embeddings`` pipeline as your pipeline in your ``config.yml`` or other configuration file:

.. literalinclude:: ../../sample_configs/config_supervised_embeddings.yml
    :language: yaml

The ``supervised_embeddings`` pipeline supports any language that can be tokenized.  By default it uses whitespace
for tokenization. You can customize the setup of this pipeline by adding or changing components. Here are the default
components that make up the ``supervised_embeddings`` pipeline:

.. code-block:: yaml

    language: "en"

    pipeline:
    - name: "WhitespaceTokenizer"
    - name: "RegexFeaturizer"
    - name: "CRFEntityExtractor"
    - name: "EntitySynonymMapper"
    - name: "CountVectorsFeaturizer"
    - name: "CountVectorsFeaturizer"
      analyzer: "char_wb"
      min_ngram: 1
      max_ngram: 4
    - name: "EmbeddingIntentClassifier"
    
So for example, if your chosen language is not whitespace-tokenized (words are not separated by spaces), you
can replace the ``WhitespaceTokenizer`` with your own tokenizer. We support a number of different :ref:`tokenizers <tokenizers>`,
or you can :ref:`create your own <custom-nlu-components>`.

The pipeline uses two instances of ``CountVectorsFeaturizer``. The first one 
featurizes text based on words. The second one featurizes text based on character 
n-grams, preserving word boundaries. We empirically found the second featurizer 
to be more powerful, but we decided to keep the first featurizer as well to make
featurization more robust.

.. _section_pretrained_embeddings_spacy_pipeline:

pretrained_embeddings_spacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use the ``pretrained_embeddings_spacy`` template:

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_spacy.yml
    :language: yaml

See :ref:`pretrained-word-vectors` for more information about loading spacy language models.
To use the components and configure them separately:

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

.. _section_mitie_pipeline:

MITIE
~~~~~

To use the MITIE pipeline, you will have to train word vectors from a corpus. Instructions can be found
:ref:`here <mitie>`. This will give you the file path to pass to the ``model`` parameter.

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie.yml
    :language: yaml

Another version of this pipeline uses MITIE's featurizer and also its multi-class classifier.
Training can be quite slow, so this is not recommended for large datasets.

.. literalinclude:: ../../sample_configs/config_pretrained_embeddings_mitie_2.yml
    :language: yaml


Custom pipelines
----------------

You don't have to use a template, you can also run a fully custom pipeline
by listing the names of the components you want to use:

.. code-block:: yaml

    pipeline:
    - name: "SpacyNLP"
    - name: "CRFEntityExtractor"
    - name: "EntitySynonymMapper"

This creates a pipeline that only does entity recognition, but no
intent classification. So Rasa NLU will not predict any intents.
You can find the details of each component in :ref:`components`.

If you want to use custom components in your pipeline, see :ref:`custom-nlu-components`.
