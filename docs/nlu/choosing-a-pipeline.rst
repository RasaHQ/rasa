:desc: Set up a pipeline of components. a
 a
.. _choosing-a-pipeline: a
 a
Choosing a Pipeline a
=================== a
 a
.. edit-link:: a
 a
In Rasa Open Source, incoming messages are processed by a sequence of components. a
These components are executed one after another in a so-called processing ``pipeline`` defined in your ``config.yml``. a
Choosing an NLU pipeline allows you to customize your model and finetune it on your dataset. a
 a
 a
.. contents:: a
   :local: a
   :depth: 2 a
 a
.. note:: a
    With Rasa 1.8.0 we updated some components and deprecated all existing pipeline templates. a
    However, **any of the old terminology will still behave the same way as it did before**! a
 a
.. warning:: a
    We deprecated all existing pipeline templates (e.g. ``supervised_embeddings``). Please list any a
    components you want to use directly in the configuration file. See a
    :ref:`how-to-choose-a-pipeline` for recommended starting configurations, or a
    :ref:`pipeline-templates` for more information. a
 a
 a
.. _how-to-choose-a-pipeline: a
 a
How to Choose a Pipeline a
------------------------ a
 a
The Short Answer a
**************** a
 a
If your training data is in English, a good starting point is the following pipeline: a
 a
    .. literalinclude:: ../../data/configs_for_docs/default_english_config.yml a
        :language: yaml a
 a
If your training data is not in English, start with the following pipeline: a
 a
    .. literalinclude:: ../../data/configs_for_docs/default_config.yml a
        :language: yaml a
 a
 a
A Longer Answer a
*************** a
 a
.. _recommended-pipeline-english: a
 a
We recommend using following pipeline, if your training data is in English: a
 a
    .. literalinclude:: ../../data/configs_for_docs/default_english_config.yml a
        :language: yaml a
 a
The pipeline contains the :ref:`ConveRTFeaturizer` that provides pre-trained word embeddings of the user utterance. a
Pre-trained word embeddings are helpful as they already encode some kind of linguistic knowledge. a
For example, if you have a sentence like "I want to buy apples" in your training data, and Rasa is asked to predict a
the intent for "get pears", your model already knows that the words "apples" and "pears" are very similar. a
This is especially useful if you don’t have enough training data. a
The advantage of the :ref:`ConveRTFeaturizer` is that it doesn't treat each word of the user message independently, but a
creates a contextual vector representation for the complete sentence. a
However, ``ConveRT`` is only available in English. a
 a
 a
.. _recommended-pipeline-pretrained-non-english: a
 a
If your training data is not in English, but you still want to use pre-trained word embeddings, we recommend using a
the following pipeline: a
 a
    .. literalinclude:: ../../data/configs_for_docs/default_spacy_config.yml a
        :language: yaml a
 a
It uses the :ref:`SpacyFeaturizer` instead of the :ref:`ConveRTFeaturizer`. a
:ref:`SpacyFeaturizer` provides pre-trained word embeddings from either GloVe or fastText in many different languages a
(see :ref:`pretrained-word-vectors`). a
 a
 a
.. _recommended-pipeline-non-english: a
 a
If you don't use any pre-trained word embeddings inside your pipeline, you are not bound to a specific language a
and can train your model to be more domain specific. a
If there are no word embeddings for your language or you have very domain specific terminology, a
we recommend using the following pipeline: a
 a
    .. literalinclude:: ../../data/configs_for_docs/default_config.yml a
        :language: yaml a
 a
.. note:: a
    We encourage everyone to define their own pipeline by listing the names of the components you want to use. a
    You can find the details of each component in :ref:`components`. a
    If you want to use custom components in your pipeline, see :ref:`custom-nlu-components`. a
 a
Choosing the Right Components a
***************************** a
 a
There are components for entity extraction, for intent classification, response selection, a
pre-processing, and others. You can learn more about any specific component on the :ref:`components` page. a
If you want to add your own component, for example to run a spell-check or to a
do sentiment analysis, check out :ref:`custom-nlu-components`. a
 a
A pipeline usually consists of three main parts: a
 a
.. contents:: a
   :local: a
   :depth: 1 a
 a
 a
Tokenization a
~~~~~~~~~~~~ a
 a
For tokenization of English input, we recommend the :ref:`ConveRTTokenizer`. a
You can process other whitespace-tokenized (words are separated by spaces) languages a
with the :ref:`WhitespaceTokenizer`. If your language is not whitespace-tokenized, you should use a different tokenizer. a
We support a number of different :ref:`tokenizers <tokenizers>`, or you can a
create your own :ref:`custom tokenizer <custom-nlu-components>`. a
 a
.. note:: a
    Some components further down the pipeline may require a specific tokenizer. You can find those requirements a
    on the individual components in :ref:`components`. If a required component is missing inside the pipeline, an a
    error will be thrown. a
 a
 a
Featurization a
~~~~~~~~~~~~~ a
 a
You need to decide whether to use components that provide pre-trained word embeddings or not. We recommend in cases a
of small amounts of training data to start with pre-trained word embeddings. Once you have a larger amount of data a
and ensure that most relevant words will be in your data and therefore will have a word embedding, supervised a
embeddings, which learn word meanings directly from your training data, can make your model more specific to your domain. a
If you can't find a pre-trained model for your language, you should use supervised embeddings. a
 a
.. contents:: a
   :local: a
 a
Pre-trained Embeddings a
^^^^^^^^^^^^^^^^^^^^^^ a
 a
The advantage of using pre-trained word embeddings in your pipeline is that if you have a training example like: a
"I want to buy apples", and Rasa is asked to predict the intent for "get pears", your model already knows that the a
words "apples" and "pears" are very similar. This is especially useful if you don't have enough training data. a
We support a few components that provide pre-trained word embeddings: a
 a
1. :ref:`MitieFeaturizer` a
2. :ref:`SpacyFeaturizer` a
3. :ref:`ConveRTFeaturizer` a
4. :ref:`LanguageModelFeaturizer` a
 a
If your training data is in English, we recommend using the :ref:`ConveRTFeaturizer`. a
The advantage of the :ref:`ConveRTFeaturizer` is that it doesn't treat each word of the user message independently, but a
creates a contextual vector representation for the complete sentence. For example, if you a
have a training example, like: "Can I book a car?", and Rasa is asked to predict the intent for "I need a ride from a
my place", since the contextual vector representation for both examples are already very similar, the intent classified a
for both is highly likely to be the same. This is also useful if you don't have enough training data. a
 a
An alternative to :ref:`ConveRTFeaturizer` is the :ref:`LanguageModelFeaturizer` which uses pre-trained language a
models such as BERT, GPT-2, etc. to extract similar contextual vector representations for the complete sentence. See a
:ref:`HFTransformersNLP` for a full list of supported language models. a
 a
If your training data is not in English you can also use a different variant of a language model which a
is pre-trained in the language specific to your training data. a
For example, there are chinese (``bert-base-chinese``) and japanese (``bert-base-japanese``) variants of the BERT model. a
A full list of different variants of a
these language models is available in the a
`official documentation of the Transformers library <https://huggingface.co/transformers/pretrained_models.html>`_. a
 a
:ref:`SpacyFeaturizer` also provides word embeddings in many different languages (see :ref:`pretrained-word-vectors`), a
so you can use this as another alternative, depending on the language of your training data. a
 a
Supervised Embeddings a
^^^^^^^^^^^^^^^^^^^^^ a
 a
If you don't use any pre-trained word embeddings inside your pipeline, you are not bound to a specific language a
and can train your model to be more domain specific. For example, in general English, the word "balance" is closely a
related to "symmetry", but very different to the word "cash". In a banking domain, "balance" and "cash" are closely a
related and you'd like your model to capture that. a
You should only use featurizers from the category :ref:`sparse featurizers <text-featurizers>`, such as a
:ref:`CountVectorsFeaturizer`, :ref:`RegexFeaturizer` or :ref:`LexicalSyntacticFeaturizer`, if you don't want to use a
pre-trained word embeddings. a
 a
 a
Entity Recognition / Intent Classification / Response Selectors a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
Depending on your data you may want to only perform intent classification, entity recognition or response selection. a
Or you might want to combine multiple of those tasks. a
We support several components for each of the tasks. All of them are listed in :ref:`components`. a
We recommend using :ref:`diet-classifier` for intent classification and entity recognition a
and :ref:`response-selector` for response selection. a
 a
 a
Multi-Intent Classification a
*************************** a
 a
You can use Rasa Open Source components to split intents into multiple labels. For example, you can predict a
multiple intents (``thank+goodbye``) or model hierarchical intent structure (``feedback+positive`` being more similar a
to ``feedback+negative`` than ``chitchat``). a
To do this, you need to use the :ref:`diet-classifier` in your pipeline. a
You'll also need to define these flags in whichever tokenizer you are using: a
 a
    - ``intent_tokenization_flag``: Set it to ``True``, so that intent labels are tokenized. a
    - ``intent_split_symbol``: Set it to the delimiter string that splits the intent labels. In this case ``+``, default ``_``. a
 a
Read a `tutorial <https://blog.rasa.com/how-to-handle-multiple-intents-per-input-using-rasa-nlu-tensorflow-pipeline/>`__ a
on how to use multiple intents in Rasa. a
 a
Here's an example configuration: a
 a
    .. code-block:: yaml a
 a
        language: "en" a
 a
        pipeline: a
        - name: "WhitespaceTokenizer" a
          intent_tokenization_flag: True a
          intent_split_symbol: "_" a
        - name: "CountVectorsFeaturizer" a
        - name: "DIETClassifier" a
 a
 a
Comparing Pipelines a
------------------- a
 a
Rasa gives you the tools to compare the performance of multiple pipelines on your data directly. a
See :ref:`comparing-nlu-pipelines` for more information. a
 a
.. note:: a
 a
    Intent classification is independent of entity extraction. So sometimes a
    NLU will get the intent right but entities wrong, or the other way around. a
    You need to provide enough data for both intents and entities. a
 a
 a
Handling Class Imbalance a
------------------------ a
 a
Classification algorithms often do not perform well if there is a large `class imbalance`, a
for example if you have a lot of training data for some intents and very little training data for others. a
To mitigate this problem, you can use a ``balanced`` batching strategy. a
This algorithm ensures that all classes are represented in every batch, or at least in a
as many subsequent batches as possible, still mimicking the fact that some classes are more frequent than others. a
Balanced batching is used by default. In order to turn it off and use a classic batching strategy include a
``batch_strategy: sequence`` in your config file. a
 a
    .. code-block:: yaml a
 a
        language: "en" a
 a
        pipeline: a
        # - ... other components a
        - name: "DIETClassifier" a
          batch_strategy: sequence a
 a
 a
.. _component-lifecycle: a
 a
Component Lifecycle a
------------------- a
 a
Each component processes an input and/or creates an output. The order of the components is determined by a
the order they are listed in the ``config.yml``; the output of a component can be used by any other component that a
comes after it in the pipeline. Some components only produce information used by other components a
in the pipeline. Other components produce ``output`` attributes that are returned after a
the processing has finished. a
 a
For example, for the sentence ``"I am looking for Chinese food"``, the output is: a
 a
    .. code-block:: json a
 a
        { a
            "text": "I am looking for Chinese food", a
            "entities": [ a
                { a
                    "start": 8, a
                    "end": 15, a
                    "value": "chinese", a
                    "entity": "cuisine", a
                    "extractor": "DIETClassifier", a
                    "confidence": 0.864 a
                } a
            ], a
            "intent": {"confidence": 0.6485910906220309, "name": "restaurant_search"}, a
            "intent_ranking": [ a
                {"confidence": 0.6485910906220309, "name": "restaurant_search"}, a
                {"confidence": 0.1416153159565678, "name": "affirm"} a
            ] a
        } a
 a
This is created as a combination of the results of the different components in the following pipeline: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
          - name: WhitespaceTokenizer a
          - name: RegexFeaturizer a
          - name: LexicalSyntacticFeaturizer a
          - name: CountVectorsFeaturizer a
          - name: CountVectorsFeaturizer a
            analyzer: "char_wb" a
            min_ngram: 1 a
            max_ngram: 4 a
          - name: DIETClassifier a
          - name: EntitySynonymMapper a
          - name: ResponseSelector a
 a
For example, the ``entities`` attribute here is created by the ``DIETClassifier`` component. a
 a
Every component can implement several methods from the ``Component`` base class; in a pipeline these different methods a
will be called in a specific order. Assuming we added the following pipeline to our ``config.yml``: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
          - name: "Component A" a
          - name: "Component B" a
          - name: "Last Component" a
 a
The image below shows the call order during the training of this pipeline: a
 a
.. image:: /_static/images/component_lifecycle.png a
 a
Before the first component is created using the ``create`` function, a so a
called ``context`` is created (which is nothing more than a python dict). a
This context is used to pass information between the components. For example, a
one component can calculate feature vectors for the training data, store a
that within the context and another component can retrieve these feature a
vectors from the context and do intent classification. a
 a
Initially the context is filled with all configuration values. The arrows a
in the image show the call order and visualize the path of the passed a
context. After all components are trained and persisted, the a
final context dictionary is used to persist the model's metadata. a
 a
.. _pipeline-templates: a
 a
Pipeline Templates (deprecated) a
------------------------------- a
 a
A template is just a shortcut for a full list of components. For example, this pipeline template: a
 a
    .. code-block:: yaml a
 a
        language: "en" a
        pipeline: "pretrained_embeddings_spacy" a
 a
is equivalent to this pipeline: a
 a
    .. code-block:: yaml a
 a
        language: "en" a
        pipeline: a
        - name: "SpacyNLP" a
        - name: "SpacyTokenizer" a
        - name: "SpacyFeaturizer" a
        - name: "RegexFeaturizer" a
        - name: "CRFEntityExtractor" a
        - name: "EntitySynonymMapper" a
        - name: "SklearnIntentClassifier" a
 a
Pipeline templates are deprecated as of Rasa 1.8. To find sensible configurations to get started, a
check out :ref:`how-to-choose-a-pipeline`. For more information about a deprecated pipeline template, a
expand it below. a
 a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            ``pretrained_embeddings_spacy`` a
 a
        .. _section_pretrained_embeddings_spacy_pipeline: a
 a
        The advantage of ``pretrained_embeddings_spacy`` pipeline is that if you have a training example like: a
        "I want to buy apples", and Rasa is asked to predict the intent for "get pears", your model a
        already knows that the words "apples" and "pears" are very similar. This is especially useful a
        if you don't have enough training data. a
 a
        See :ref:`pretrained-word-vectors` for more information about loading spacy language models. a
        To use the components and configure them separately: a
 a
            .. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_spacy_config.yml a
                :language: yaml a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            ``pretrained_embeddings_convert`` a
 a
        .. _section_pretrained_embeddings_convert_pipeline: a
 a
            .. note:: a
                Since ``ConveRT`` model is trained only on an **English** corpus of conversations, this pipeline should only a
                be used if your training data is in English language. a
 a
        This pipeline uses the `ConveRT <https://github.com/PolyAI-LDN/polyai-models>`_ model to extract a vector representation of a
        a sentence and feeds them to the ``DIETClassifier`` for intent classification. a
        The advantage of using the ``pretrained_embeddings_convert`` pipeline is that it doesn't treat each word of the user a
        message independently, but creates a contextual vector representation for the complete sentence. For example, if you a
        have a training example, like: "can I book a car?", and Rasa is asked to predict the intent for "I need a ride from a
        my place", since the contextual vector representation for both examples are already very similar, the intent classified a
        for both is highly likely to be the same. This is also useful if you don't have enough training data. a
 a
            .. note:: a
                To use ``pretrained_embeddings_convert`` pipeline, you should install Rasa with ``pip install rasa[convert]``. a
                Please also note that one of the dependencies(``tensorflow-text``) is currently only supported on Linux a
                platforms. a
 a
        To use the components and configure them separately: a
 a
        .. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_convert_config.yml a
            :language: yaml a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            ``supervised_embeddings`` a
 a
        .. _section_supervised_embeddings_pipeline: a
 a
        The advantage of the ``supervised_embeddings`` pipeline is that your word vectors will be customised a
        for your domain. For example, in general English, the word "balance" is closely related to "symmetry", a
        but very different to the word "cash". In a banking domain, "balance" and "cash" are closely related a
        and you'd like your model to capture that. This pipeline doesn't use a language-specific model, a
        so it will work with any language that you can tokenize (on whitespace or using a custom tokenizer). a
 a
        You can read more about this topic `in this blog post <https://medium.com/rasa-blog/supervised-word-vectors-from-scratch-in-rasa-nlu-6daf794efcd8>`__ . a
 a
        The ``supervised_embeddings`` pipeline supports any language that can be whitespace tokenized. By default it uses a
        whitespace for tokenization. You can customize the setup of this pipeline by adding or changing components. Here are a
        the default components that make up the ``supervised_embeddings`` pipeline: a
 a
            .. literalinclude:: ../../data/configs_for_docs/supervised_embeddings_config.yml a
                :language: yaml a
 a
        So for example, if your chosen language is not whitespace-tokenized (words are not separated by spaces), you a
        can replace the ``WhitespaceTokenizer`` with your own tokenizer. We support a number of different :ref:`tokenizers <tokenizers>`, a
        or you can :ref:`create your own <custom-nlu-components>`. a
 a
        The pipeline uses two instances of ``CountVectorsFeaturizer``. The first one a
        featurizes text based on words. The second one featurizes text based on character a
        n-grams, preserving word boundaries. We empirically found the second featurizer a
        to be more powerful, but we decided to keep the first featurizer as well to make a
        featurization more robust. a
 a
    .. _section_mitie_pipeline: a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            ``MITIE pipeline`` a
 a
        You can also use MITIE as a source of word vectors in your pipeline. a
        The MITIE backend performs well for small datasets, but training can take very long if you have more than a couple a
        of hundred examples. a
 a
        However, we do not recommend that you use it as mitie support is likely to be deprecated in a future release. a
 a
        To use the MITIE pipeline, you will have to train word vectors from a corpus. Instructions can be found a
        :ref:`here <mitie>`. This will give you the file path to pass to the ``model`` parameter. a
 a
            .. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_mitie_config_1.yml a
                :language: yaml a
 a
        Another version of this pipeline uses MITIE's featurizer and also its multi-class classifier. a
        Training can be quite slow, so this is not recommended for large datasets. a
 a
            .. literalinclude:: ../../data/configs_for_docs/pretrained_embeddings_mitie_config_2.yml a
                :language: yaml a
 a