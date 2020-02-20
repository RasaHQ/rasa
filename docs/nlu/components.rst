:desc: Customize the components and parameters of Rasa's Machine Learning based
       Natural Language Understanding pipeline

.. _components:

Components
==========

.. edit-link::

.. note::
   For clarity, we have renamed the pre-defined pipelines to reflect
   what they *do* rather than which libraries they use as of Rasa NLU
   0.15. The ``tensorflow_embedding`` pipeline is now called
   ``supervised_embeddings``, and ``spacy_sklearn`` is now known as
   ``pretrained_embeddings_spacy``. Please update your code if you are using these.

.. note::
    We deprecated all pre-defined pipeline templates. Take a look at :ref:`choosing-a-pipeline`
    to decide on what components you should use in your configuration file.

This is a reference of the configuration options for every built-in
component in Rasa NLU. If you want to build a custom component, check
out :ref:`custom-nlu-components`.

.. contents::
   :local:


Word Vector Sources
-------------------

.. _MitieNLP:

MitieNLP
~~~~~~~~

:Short: MITIE initializer
:Outputs: Nothing
:Requires: Nothing
:Description:
    Initializes mitie structures. Every mitie component relies on this,
    hence this should be put at the beginning
    of every pipeline that uses any mitie components.
:Configuration:
    The MITIE library needs a language model file, that **must** be specified in
    the configuration:

    .. code-block:: yaml

        pipeline:
        - name: "MitieNLP"
          # language model to load
          model: "data/total_word_feature_extractor.dat"

    For more information where to get that file from, head over to
    :ref:`installing MITIE <install-mitie>`.

.. _SpacyNLP:

SpacyNLP
~~~~~~~~

:Short: spaCy language initializer
:Outputs: Nothing
:Requires: Nothing
:Description:
    Initializes spaCy structures. Every spaCy component relies on this, hence this should be put at the beginning
    of every pipeline that uses any spaCy components.
:Configuration:
    Language model, default will use the configured language.
    If the spaCy model to be used has a name that is different from the language tag (``"en"``, ``"de"``, etc.),
    the model name can be specified using this configuration variable. The name will be passed to ``spacy.load(name)``.

    .. code-block:: yaml

        pipeline:
        - name: "SpacyNLP"
          # language model to load
          model: "en_core_web_md"

          # when retrieving word vectors, this will decide if the casing
          # of the word is relevant. E.g. `hello` and `Hello` will
          # retrieve the same vector, if set to `false`. For some
          # applications and models it makes sense to differentiate
          # between these two words, therefore setting this to `true`.
          case_sensitive: false

    For more information on how to obtain the spaCy models, head over to
    :ref:`installing SpaCy <install-spacy>`.

.. _HFTransformersNLP:

HFTransformersNLP
~~~~~~~~~~~~~~~~~

:Short: HuggingFace's Transformers based pre-trained language model initializer
:Outputs: Nothing
:Requires: Nothing
:Description:
    Initializes specified pre-trained language model from HuggingFace's `Transformers library
    <https://huggingface.co/transformers/>`__.  The component applies language model specific tokenization and
    featurization to compute sequence and sentence level representations for each example in the training data.
    Include :ref:`LanguageModelTokenizer` and :ref:`LanguageModelFeaturizer` to utilize the output of this
    component for downstream NLU models.
:Configuration:
    .. code-block:: yaml

        pipeline:
          - name: HFTransformersNLP

            # Name of the language model to use
            model_name: "bert"

            # Shortcut name to specify architecture variation of the above model. Full list of supported architectures
            # can be found at https://huggingface.co/transformers/pretrained_models.html . If left empty, it uses the
            # default model architecture that original transformers library loads
            model_weights: "bert-base-uncased"

        #    +----------------+--------------+-------------------------+
        #    | Language Model | Parameter    | Default value for       |
        #    |                | "model_name" | "model_weights"         |
        #    +----------------+--------------+-------------------------+
        #    | BERT           | bert         | bert-base-uncased       |
        #    +----------------+--------------+-------------------------+
        #    | GPT            | gpt          | openai-gpt              |
        #    +----------------+--------------+-------------------------+
        #    | GPT-2          | gpt2         | gpt2                    |
        #    +----------------+--------------+-------------------------+
        #    | XLNet          | xlnet        | xlnet-base-cased        |
        #    +----------------+--------------+-------------------------+
        #    | DistilBERT     | distilbert   | distilbert-base-uncased |
        #    +----------------+--------------+-------------------------+
        #    | RoBERTa        | roberta      | roberta-base            |
        #    +----------------+--------------+-------------------------+



.. _tokenizers:

Tokenizers
----------

Tokenizers split text into tokens.
If you want to split intents into multiple labels, e.g. for predicting multiple intents or for
modeling hierarchical intent structure, use these flags with any tokenizer:

- ``intent_tokenization_flag`` indicates whether to tokenize intent labels or not. By default this flag is set to
  ``False``, intent will not be tokenized.
- ``intent_split_symbol`` sets the delimiter string to split the intent labels, default is underscore
  (``_``).

    .. note:: All tokenizer add an additional token ``__CLS__`` to the end of the list of tokens when tokenizing
              text and responses.

WhitespaceTokenizer
~~~~~~~~~~~~~~~~~~~

:Short: Tokenizer using whitespaces as a separator
:Outputs: ``tokens`` for texts, responses (if present), and intents (if specified)
:Requires: Nothing
:Description:
    Creates a token for every whitespace separated character sequence.
:Configuration:
    Make the tokenizer not case sensitive by adding the ``case_sensitive: False`` option.
    Default being ``case_sensitive: True``.

    .. code-block:: yaml

        pipeline:
        - name: "WhitespaceTokenizer"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"
          # Text will be tokenized with case sensitive as default
          "case_sensitive": True


JiebaTokenizer
~~~~~~~~~~~~~~

:Short: Tokenizer using Jieba for Chinese language
:Outputs: ``tokens`` for texts, responses (if present), and intents (if specified)
:Requires: Nothing
:Description:
    Creates tokens using the Jieba tokenizer specifically for Chinese
    language. For language other than Chinese, Jieba will work as
    ``WhitespaceTokenizer``.

    .. note::
        To use ``JiebaTokenizer`` you need to install Jieba with ``pip install jieba``.

:Configuration:
    User's custom dictionary files can be auto loaded by specifying the files' directory path via ``dictionary_path``.
    If the ``dictionary_path`` is ``None`` (the default), then no custom dictionary will be used.

    .. code-block:: yaml

        pipeline:
        - name: "JiebaTokenizer"
          dictionary_path: "path/to/custom/dictionary/dir"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"


MitieTokenizer
~~~~~~~~~~~~~~

:Short: Tokenizer using MITIE
:Outputs: ``tokens`` for texts, responses (if present), and intents (if specified)
:Requires: :ref:`MitieNLP`
:Description: Creates tokens using the MITIE tokenizer.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "MitieTokenizer"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"

SpacyTokenizer
~~~~~~~~~~~~~~

:Short: Tokenizer using spaCy
:Outputs: ``tokens`` for texts, responses (if present), and intents (if specified)
:Requires: :ref:`SpacyNLP`
:Description:
    Creates tokens using the spaCy tokenizer.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "SpacyTokenizer"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"

.. _ConveRTTokenizer:

ConveRTTokenizer
~~~~~~~~~~~~~~~~

:Short: Tokenizer using ConveRT
:Outputs: ``tokens`` for texts, responses (if present), and intents (if specified)
:Requires: Nothing
:Description:
    Creates tokens using the ConveRT tokenizer. Must be used whenever the ``ConveRTFeaturizer`` is used.
:Configuration:
    Make the tokenizer not case sensitive by adding the ``case_sensitive: False`` option.
    Default being ``case_sensitive: True``.

    .. code-block:: yaml

        pipeline:
        - name: "ConveRTTokenizer"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"
          # Text will be tokenized with case sensitive as default
          "case_sensitive": True


.. _LanguageModelTokenizer:

LanguageModelTokenizer
~~~~~~~~~~~~~~~~~~~~~~

:Short: Tokenizer from pre-trained language models
:Outputs: ``tokens`` for texts, responses (if present), and intents (if specified)
:Requires: :ref:`HFTransformersNLP`
:Description:
    Creates tokens using the pre-trained language model specified in upstream :ref:`HFTransformersNLP` component.
    Must be used whenever the ``LanguageModelFeaturizer`` is used.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "LanguageModelTokenizer"



.. _text-featurizers:

Text Featurizers
----------------

Text featurizers are divided into two different categories: sparse featurizers and dense featurizers.
Sparse featurizers are featurizers that return feature vectors with a lot of missing values, e.g. zeros.
As those feature vectors would normally take up a lot of memory, we store them as sparse features.
Sparse features only store the values that are non zero and their positions in the vector.
Thus, we save a lot of memory and are able to train on larger datasets.

By default all featurizers will return a matrix of length (number-of-tokens x feature-dimension).
So, the returned matrix will have a feature vector for every token.
This allows us to train sequence models.
However, the additional token at the end (e.g. ``__CLS__``) contains features for the complete utterance.
This feature vector can be used in any non-sequence model.
The corresponding classifier can therefore decide what kind of features to use.

MitieFeaturizer
~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using the MITIE featurizer.
:Outputs: ``dense_features`` for texts and responses
:Requires: :ref:`MitieNLP`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response classification using the MITIE
    featurizer.

    .. note::

        NOT used by the ``MitieIntentClassifier`` component.

:Configuration:
    The sentence vector, e.g. the vector of the ``__CLS__`` token can be calculated in two different ways, either via
    mean or via max pooling. You can specify the pooling method in your configuration file with the option ``pooling``.
    The default pooling method is set to ``mean``.

    .. code-block:: yaml

        pipeline:
        - name: "MitieFeaturizer"
          # Specify what pooling operation should be used to calculate the vector of
          # the __CLS__ token. Available options: 'mean' and 'max'.
          "pooling": "mean"


SpacyFeaturizer
~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using the spaCy featurizer.
:Outputs: ``dense_features`` for texts and responses
:Requires: :ref:`SpacyNLP`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response classification using the spaCy
    featurizer.
:Configuration:
    The sentence vector, e.g. the vector of the ``__CLS__`` token can be calculated in two different ways, either via
    mean or via max pooling. You can specify the pooling method in your configuration file with the option ``pooling``.
    The default pooling method is set to ``mean``.

    .. code-block:: yaml

        pipeline:
        - name: "SpacyFeaturizer"
          # Specify what pooling operation should be used to calculate the vector of
          # the __CLS__ token. Available options: 'mean' and 'max'.
          "pooling": "mean"


ConveRTFeaturizer
~~~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using
    `ConveRT <https://github.com/PolyAI-LDN/polyai-models>`_ model.
:Outputs: ``dense_features`` for texts and responses
:Requires: :ref:`ConveRTTokenizer`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response selection.
    Uses the `default signature <https://github.com/PolyAI-LDN/polyai-models#tfhub-signatures>`_ to compute vector
    representations of input text.

    .. warning::
        Since ``ConveRT`` model is trained only on an english corpus of conversations, this featurizer should only
        be used if your training data is in english language.

    .. note::
        To use ``ConveRTFeaturizer`` you need to install additional tensorflow libraries (``tensorflow_text`` and
        ``tensorflow_hub``). You should do a pip install of Rasa with ``pip install rasa[convert]`` to install those.

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "ConveRTFeaturizer"


.. _LanguageModelFeaturizer:

LanguageModelFeaturizer
~~~~~~~~~~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using a pre-trained language model.
:Outputs: ``dense_features`` for texts and responses
:Requires: :ref:`HFTransformersNLP`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response selection.
    Uses the pre-trained language model specified in upstream :ref:`HFTransformersNLP` component to compute vector
    representations of input text.

    .. warning::
        Please make sure that you use a language model which is pre-trained on the same language corpus as that of your
        training data.

:Configuration:

    Include ``HFTransformersNLP`` component before this component. Also, use :ref:`LanguageModelTokenizer` to ensure
    tokens are correctly set for all components throughout the pipeline.

    .. code-block:: yaml

        pipeline:
        - name: "LanguageModelFeaturizer"


RegexFeaturizer
~~~~~~~~~~~~~~~

:Short: Creates a vector representation of user message using regular expressions.
:Outputs: ``sparse_features`` for texts and ``tokens.pattern``
:Requires: ``tokens``
:Type: Sparse featurizer
:Description:
    Creates features for entity extraction and intent classification.
    During training ``RegexFeaturizer`` creates a list of `regular expressions` defined in the training
    data format.
    For each regex, a feature will be set marking whether this expression was found in the input, which will later
    be fed into intent classifier / entity extractor to simplify classification (assuming the classifier has learned
    during the training phase, that this set feature indicates a certain intent / entity).
    Regex features for entity extraction are currently only supported by the ``CRFEntityExtractor`` and the
    ``DIETClassifier`` components!

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "RegexFeaturizer"

CountVectorsFeaturizer
~~~~~~~~~~~~~~~~~~~~~~

:Short: Creates bag-of-words representation of user messages, intents, and responses.
:Outputs: ``sparse_features`` for texts, intents, and responses
:Requires: ``tokens``
:Type: Sparse featurizer
:Description:
    Creates features for intent classification and response selection.
    Creates bag-of-words representation of user message, intent, and response using
    `sklearn's CountVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_.
    All tokens which consist only of digits (e.g. 123 and 99 but not a123d) will be assigned to the same feature.

    .. note::
        If the words in the model language cannot be split by whitespace,
        a language-specific tokenizer is required in the pipeline before this component
        (e.g. using ``JiebaTokenizer`` for Chinese).

:Configuration:
    See `sklearn's CountVectorizer docs <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
    for detailed description of the configuration parameters.

    This featurizer can be configured to use word or character n-grams, using ``analyzer`` config parameter.
    By default ``analyzer`` is set to ``word`` so word token counts are used as features.
    If you want to use character n-grams, set ``analyzer`` to ``char`` or ``char_wb``.

    .. note::
        Option ‘char_wb’ creates character n-grams only from text inside word boundaries;
        n-grams at the edges of words are padded with space.
        This option can be used to create `Subword Semantic Hashing <https://arxiv.org/abs/1810.07150>`_

    .. note::
        For character n-grams do not forget to increase ``min_ngram`` and ``max_ngram`` parameters.
        Otherwise the vocabulary will contain only single letters

    Handling Out-Of-Vacabulary (OOV) words:

        .. note:: Enabled only if ``analyzer`` is ``word``.

        Since the training is performed on limited vocabulary data, it cannot be guaranteed that during prediction
        an algorithm will not encounter an unknown word (a word that were not seen during training).
        In order to teach an algorithm how to treat unknown words, some words in training data can be substituted
        by generic word ``OOV_token``.
        In this case during prediction all unknown words will be treated as this generic word ``OOV_token``.

        For example, one might create separate intent ``outofscope`` in the training data containing messages of
        different number of ``OOV_token`` s and maybe some additional general words.
        Then an algorithm will likely classify a message with unknown words as this intent ``outofscope``.

        .. note::
            This featurizer creates a bag-of-words representation by **counting** words,
            so the number of ``OOV_token`` in the sentence might be important.

            - ``OOV_token`` set a keyword for unseen words; if training data contains ``OOV_token`` as words in some
              messages, during prediction the words that were not seen during training will be substituted with
              provided ``OOV_token``; if ``OOV_token=None`` (default behaviour) words that were not seen during
              training will be ignored during prediction time;
            - ``OOV_words`` set a list of words to be treated as ``OOV_token`` during training; if a list of words
              that should be treated as Out-Of-Vacabulary is known, it can be set to ``OOV_words`` instead of manually
              changing it in trainig data or using custom preprocessor.

        .. note::
            Providing ``OOV_words`` is optional, training data can contain ``OOV_token`` input manually or by custom
            additional preprocessor.
            Unseen words will be substituted with ``OOV_token`` **only** if this token is present in the training
            data or ``OOV_words`` list is provided.

    Sharing Vocabulary between user message and labels:

        .. note:: Enabled only if ``use_shared_vocab`` is ``True``

        Build a common vocabulary set between tokens in labels and user message.

    .. code-block:: yaml

        pipeline:
        - name: "CountVectorsFeaturizer"
          # whether to use a shared vocab
          "use_shared_vocab": False,
          # whether to use word or character n-grams
          # 'char_wb' creates character n-grams only inside word boundaries
          # n-grams at the edges of words are padded with space.
          analyzer: 'word'  # use 'char' or 'char_wb' for character
          # the parameters are taken from
          # sklearn's CountVectorizer
          # regular expression for tokens
          token_pattern: r'(?u)\b\w\w+\b'
          # remove accents during the preprocessing step
          strip_accents: None  # {'ascii', 'unicode', None}
          # list of stop words
          stop_words: None  # string {'english'}, list, or None (default)
          # min document frequency of a word to add to vocabulary
          # float - the parameter represents a proportion of documents
          # integer - absolute counts
          min_df: 1  # float in range [0.0, 1.0] or int
          # max document frequency of a word to add to vocabulary
          # float - the parameter represents a proportion of documents
          # integer - absolute counts
          max_df: 1.0  # float in range [0.0, 1.0] or int
          # set ngram range
          min_ngram: 1  # int
          max_ngram: 1  # int
          # limit vocabulary size
          max_features: None  # int or None
          # if convert all characters to lowercase
          lowercase: true  # bool
          # handling Out-Of-Vacabulary (OOV) words
          # will be converted to lowercase if lowercase is true
          OOV_token: None  # string or None
          OOV_words: []  # list of strings

.. _LexicalSyntacticFeaturizer:

LexicalSyntacticFeaturizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Creates lexical and syntactic features for user message to support entity extraction.
:Outputs: ``sparse_features`` for texts
:Requires: ``tokens``
:Type: Sparse featurizer
:Description:
    Creates features for entity extraction.
    Moves with a sliding window over every token in the user message and creates features according to the
    configuration (see below).
:Configuration:
    You need to configure what kind of lexical and syntactic features the featurizer should extract.
    The following features are available:

    ==============  =============================================================================================
    Feature Name    Description
    ==============  =============================================================================================
    BOS             Checks if the token is at the beginning of the sentence.
    EOS             Checks if the token is at the end of the sentence.
    low             Checks if the token is lower case.
    upper           Checks if the token is upper case.
    title           Checks if the token starts with an uppercase character and all remaining characters are
                    lowercased.
    digit           Checks if the token contains just digits.
    prefix5         Take the first five characters of the token.
    prefix2         Take the first two characters of the token.
    suffix5         Take the last five characters of the token.
    suffix3         Take the last three characters of the token.
    suffix2         Take the last two characters of the token.
    suffix1         Take the last character of the token.
    pos             Take the Part-of-Speech tag of the token (spaCy required).
    pos2            Take the first two characters of the Part-of-Speech tag of the token (spaCy required).
    ==============  =============================================================================================

    As the featurizer is moving over the tokens in a user message with a sliding window, you can define features for
    previous tokens, the current token, and the next tokens in the sliding window.
    You define the features as [before, token, after] array.
    If you, for example, want to define features for the token before, the current token, and the token after,
    your features configuration could look like this:

    .. code-block:: yaml

        pipeline:
        - name: "LexicalSyntacticFeaturizer":
          "features": [
            ["low", "title", "upper"],
            [
              "BOS",
              "EOS",
              "low",
              "prefix5",
              "prefix2",
              "suffix5",
              "suffix3",
              "suffix2",
              "upper",
              "title",
              "digit",
            ],
            ["low", "title", "upper"],
          ]

    This configuration is also the default configuration.

    .. note:: If you want to make use of ``pos`` or ``pos2`` you need to add ``SpacyTokenizer`` to your pipeline.


Intent Classifiers
------------------

Intent classifiers assign one of the intents defined in the domain file to incoming user messages.

MitieIntentClassifier
~~~~~~~~~~~~~~~~~~~~~

:Short:
    MITIE intent classifier (using a
    `text categorizer <https://github.com/mit-nlp/MITIE/blob/master/examples/python/text_categorizer_pure_model.py>`_)
:Outputs: ``intent``
:Requires: ``tokens`` for user message
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.98343}
        }

:Description:
    This classifier uses MITIE to perform intent classification. The underlying classifier
    is using a multi-class linear SVM with a sparse linear kernel (see
    `MITIE trainer code <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/text_categorizer_trainer.cpp#L222>`_).

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "MitieIntentClassifier"

SklearnIntentClassifier
~~~~~~~~~~~~~~~~~~~~~~~

:Short: Sklearn intent classifier
:Outputs: ``intent`` and ``intent_ranking``
:Requires: ``dense_features`` for user message
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
    The sklearn intent classifier trains a linear SVM which gets optimized using a grid search. It also provides
    rankings of the labels that did not "win". The ``SklearnIntentClassifier`` needs to be preceded by a dense
    featurizer in the pipeline. This dense featurizer creates the features used for the classification.

:Configuration:
    During the training of the SVM a hyperparameter search is run to
    find the best parameter set. In the config, you can specify the parameters
    that will get tried.

    .. code-block:: yaml

        pipeline:
        - name: "SklearnIntentClassifier"
          # Specifies the list of regularization values to
          # cross-validate over for C-SVM.
          # This is used with the ``kernel`` hyperparameter in GridSearchCV.
          C: [1, 2, 5, 10, 20, 100]
          # Specifies the kernel to use with C-SVM.
          # This is used with the ``C`` hyperparameter in GridSearchCV.
          kernels: ["linear"]

EmbeddingIntentClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Dual Intent Entity Transformer used for intent classification
:Outputs: ``intent`` and ``intent_ranking``
:Requires: ``dense_features`` and/or ``sparse_features`` for user message and intent (optional)
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
    The ``EmbeddingIntentClassifier`` embeds user inputs and intent labels into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    This algorithm is based on `StarSpace <https://arxiv.org/abs/1709.03856>`_.
    However, in this implementation the loss function is slightly different and
    additional hidden layers are added together with dropout.
    This algorithm also provides similarity rankings of the labels that did not "win".

    .. note:: If during prediction time a message contains **only** words unseen during training,
              and no Out-Of-Vacabulary preprocessor was used,
              empty intent ``None`` is predicted with confidence ``0.0``.

    .. warning::
        ``EmbeddingIntentClassifier`` is deprecated and should be replaced by ``DIETClassifier``. See
        `migration guide <https://rasa.com/docs/rasa/migration-guide/#rasa-1-7-to-rasa-1-8>`_ for more details.

:Configuration:

    The following hyperparameters can be set:

        - neural network's architecture:

            - ``hidden_layers_sizes.text`` sets a list of hidden layer sizes before
              the embedding layer for user inputs, the number of hidden layers
              is equal to the length of the list.
            - ``hidden_layers_sizes.label`` sets a list of hidden layer sizes before
              the embedding layer for intent labels, the number of hidden layers
              is equal to the length of the list.
            - ``share_hidden_layers`` if set to True, shares the hidden layers between user inputs and intent label.

        - training:

            - ``batch_size`` sets the number of training examples in one
              forward/backward pass, the higher the batch size, the more
              memory space you'll need.
            - ``batch_strategy`` sets the type of batching strategy,
              it should be either ``sequence`` or ``balanced``.
            - ``epochs`` sets the number of times the algorithm will see
              training data, where one ``epoch`` equals one forward pass and
              one backward pass of all the training examples.
            - ``random_seed`` if set you will get reproducible
              training results for the same inputs.
            - ``learning_rate`` sets the initial learning rate of the optimizer.

        - embedding:

            - ``dense_dimension.text`` sets the dense dimensions for user inputs to use for sparse
              tensors if no dense features are present.
            - ``dense_dimension.label`` sets the dense dimensions for intent labels to use for sparse
              tensors if no dense features are present.
            - ``embedding_dimension`` sets the dimension of embedding space.
            - ``number_of_negative_examples`` sets the number of incorrect intent labels.
              The algorithm will minimize their similarity to the user
              input during training.
            - ``similarity_type`` sets the type of the similarity,
              it should be either ``auto``, ``cosine`` or ``inner``,
              if ``auto``, it will be set depending on ``loss_type``,
              ``inner`` for ``softmax``, ``cosine`` for ``margin``.
            - ``loss_type`` sets the type of the loss function,
              it should be either ``softmax`` or ``margin``.
            - ``ranking_length`` defines the number of top confidences over
              which to normalize ranking results if ``loss_type: "softmax"``.
              To turn off normalization set it to 0.
            - ``maximum_positive_similarity`` controls how similar the algorithm should try
              to make embedding vectors for correct intent labels,
              used only if ``loss_type`` is set to ``margin``.
            - ``maximum_negative_similarity`` controls maximum negative similarity for
              incorrect intents, used only if ``loss_type`` is set to ``margin``.
            - ``use_maximum_negative_similarity`` if ``true`` the algorithm only
              minimizes maximum similarity over incorrect intent labels,
              used only if ``loss_type`` is set to ``margin``.
            - ``scale_loss`` if ``true`` the algorithm will downscale the loss
              for examples where correct label is predicted with high confidence,
              used only if ``loss_type`` is set to ``softmax``.

        - regularization:

            - ``regularization_constant`` sets the scale of L2 regularization.
            - ``negative_margin_scale`` sets the scale of how important is to minimize
              the maximum similarity between embeddings of different intent labels.
            - ``droprate`` sets the dropout rate, it should be
              between ``0`` and ``1``, e.g. ``droprate=0.1`` would drop out ``10%`` of input units.
            - ``use_sparse_input_dropout`` specifies whether to apply dropout to sparse tensors or not.

    .. note:: For ``cosine`` similarity ``maximum_positive_similarity`` and ``maximum_negative_similarity`` should
              be between ``-1`` and ``1``.

    .. note:: There is an option to use linearly increasing batch size. The idea comes from
              `<https://arxiv.org/abs/1711.00489>`_.
              In order to do it pass a list to ``batch_size``, e.g. ``"batch_size": [64, 256]`` (default behaviour).
              If constant ``batch_size`` is required, pass an ``int``, e.g. ``"batch_size": 64``.

    .. note:: Parameter ``maximum_negative_similarity`` is set to a negative value to mimic the original
              starspace algorithm in the case ``maximum_negative_similarity = maximum_positive_similarity``
              and ``use_maximum_negative_similarity = False``.
              See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.

    Default values:

    .. code-block:: yaml

        pipeline:
        - name: "EmbeddingIntentClassifier"
            # nn architecture
            # sizes of hidden layers before the embedding layer
            # for input words and intent labels,
            # the number of hidden layers is thus equal to the length of this list
            "hidden_layers_sizes": {"text": [256, 128], "label": []}
            # Whether to share the hidden layer weights between input words and labels
            "share_hidden_layers": False
            # training parameters
            # initial and final batch sizes - batch size will be
            # linearly increased for each epoch
            "batch_size": [64, 256]
            # how to create batches
            "batch_strategy": "balanced"  # string 'sequence' or 'balanced'
            # number of epochs
            "epochs": 300
            # set random seed to any int to get reproducible results
            "random_seed": None
            # optimizer
            "learning_rate": 0.001
            # embedding parameters
            # default dense dimension used if no dense features are present
            "dense_dimension": {"text": 512, "label": 20}
            # dimension size of embedding vectors
            "embedding_dimension": 20
            # the type of the similarity
            "number_of_negative_examples": 20
            # flag if minimize only maximum similarity over incorrect actions
            "similarity_type": "auto"  # string 'auto' or 'cosine' or 'inner'
            # the type of the loss function
            "loss_type": "softmax"  # string 'softmax' or 'margin'
            # number of top intents to normalize scores for softmax loss_type
            # set to 0 to turn off normalization
            "ranking_length": 10
            # how similar the algorithm should try
            # to make embedding vectors for correct labels
            "maximum_positive_similarity": 0.8  # should be 0.0 < ... < 1.0 for 'cosine'
            # maximum negative similarity for incorrect labels
            "maximum_negative_similarity": -0.4  # should be -1.0 < ... < 1.0 for 'cosine'
            # flag: if true, only minimize the maximum similarity for incorrect labels
            "use_maximum_negative_similarity": True
            # scale loss inverse proportionally to confidence of correct prediction
            "scale_loss": True
            # regularization parameters
            # the scale of regularization
            "regularization_constant": 0.002
            # the scale of how critical the algorithm should be of minimizing the
            # maximum similarity between embeddings of different labels
            "negative_margin_scale": 0.8
            # dropout rate for rnn
            "droprate": 0.2
            # if true apply dropout to sparse tensors
            "use_sparse_input_dropout": False
            # visualization of accuracy
            # how often to calculate training accuracy
            "evaluate_every_number_of_epochs": 20  # small values may hurt performance
            # how many examples to use for calculation of training accuracy
            "evaluate_on_number_of_examples": 0  # large values may hurt performance

.. _keyword_intent_classifier:

KeywordIntentClassifier
~~~~~~~~~~~~~~~~~~~~~~~

:Short: Simple keyword matching intent classifier, intended for small, short-term projects.
:Outputs: ``intent``
:Requires: Nothing

:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 1.0}
        }

:Description:
    This classifier works by searching a message for keywords.
    The matching is case sensitive by default and searches only for exact matches of the keyword-string in the user
    message.
    The keywords for an intent are the examples of that intent in the NLU training data.
    This means the entire example is the keyword, not the individual words in the example.

    .. note:: This classifier is intended only for small projects or to get started. If
              you have few NLU training data you can use one of our pipelines
              :ref:`choosing-a-pipeline`.

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "KeywordIntentClassifier"
          case_sensitive: True

Selectors
----------

.. _response-selector:

ResponseSelector
~~~~~~~~~~~~~~~~

:Short: Response Selector
:Outputs: A dictionary with key as ``direct_response_intent`` and value containing ``response`` and ``ranking``
:Requires: ``dense_features`` and/or ``sparse_features`` for user message and response

:Output-Example:

    .. code-block:: json

        {
            "response_selector": {
              "faq": {
                "response": {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
                "ranking": [
                    {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
                    {"confidence": 0.2134543431, "name": "You can ask me about how to get started"}
                ]
              }
            }
        }

:Description:

    Response Selector component can be used to build a response retrieval model to directly predict a bot response from
    a set of candidate responses. The prediction of this model is used by :ref:`retrieval-actions`.
    It embeds user inputs and response labels into the same space and follows the exact same
    neural network architecture and optimization as the ``EmbeddingIntentClassifier``.

    .. note:: If during prediction time a message contains **only** words unseen during training,
              and no Out-Of-Vacabulary preprocessor was used,
              empty response ``None`` is predicted with confidence ``0.0``.

    .. warning::
        ``ResponseSelector`` is deprecated and should be replaced by ``DIETSelector``. See
        `migration guide <https://rasa.com/docs/rasa/migration-guide/#rasa-1-7-to-rasa-1-8>`_ for more details.

:Configuration:

    The algorithm includes all the hyperparameters that ``EmbeddingIntentClassifier`` uses.
    In addition, the component can also be configured to train a response selector for a particular retrieval intent.

        - ``retrieval_intent`` sets the name of the intent for which this response selector model is trained.

    Default values:

    .. code-block:: yaml

        pipeline:
        - name: "ResponseSelector"
            # nn architecture
            # sizes of hidden layers before the embedding layer
            # for input words and intent labels,
            # the number of hidden layers is thus equal to the length of this list
            "hidden_layers_sizes": {"text": [], "label": []}
            # Whether to share the hidden layer weights between input words and labels
            "share_hidden_layers": False
            # training parameters
            # initial and final batch sizes - batch size will be
            # linearly increased for each epoch
            "batch_size": [64, 256]
            # how to create batches
            "batch_strategy": "balanced"  # string 'sequence' or 'balanced'
            # number of epochs
            "epochs": 300
            # set random seed to any int to get reproducible results
            "random_seed": None
            # optimizer
            "learning_rate": 0.001
            # embedding parameters
            # default dense dimension used if no dense features are present
            "dense_dimension": {"text": 512, "label": 512}
            # dimension size of embedding vectors
            "embedding_dimension": 20
            # the type of the similarity
            "number_of_negative_examples": 20
            # flag if minimize only maximum similarity over incorrect actions
            "similarity_type": "auto"  # string 'auto' or 'cosine' or 'inner'
            # the type of the loss function
            "loss_type": "softmax"  # string 'softmax' or 'margin'
            # number of top intents to normalize scores for softmax loss_type
            # set to 0 to turn off normalization
            "ranking_length": 10
            # how similar the algorithm should try
            # to make embedding vectors for correct labels
            "maximum_positive_similarity": 0.8  # should be 0.0 < ... < 1.0 for 'cosine'
            # maximum negative similarity for incorrect labels
            "maximum_negative_similarity": -0.4  # should be -1.0 < ... < 1.0 for 'cosine'
            # flag: if true, only minimize the maximum similarity for incorrect labels
            "use_maximum_negative_similarity": True
            # scale loss inverse proportionally to confidence of correct prediction
            "scale_loss": True
            # regularization parameters
            # the scale of regularization
            "regularization_constant": 0.002
            # the scale of how critical the algorithm should be of minimizing the
            # maximum similarity between embeddings of different labels
            "negative_margin_scale": 0.8
            # dropout rate for rnn
            "droprate": 0.2
            # if true apply dropout to sparse tensors
            "use_sparse_input_dropout": True
            # visualization of accuracy
            # how often to calculate training accuracy
            "evaluate_every_number_of_epochs": 20  # small values may hurt performance
            # how many examples to use for calculation of training accuracy
            "evaluate_on_number_of_examples": 0  # large values may hurt performance
            # selector config
            # name of the intent for which this response selector is to be trained
            "retrieval_intent": None


.. _diet-selector:

DIETSelector
~~~~~~~~~~~~~~~~

:Short: DIET Selector
:Outputs: A dictionary with key as ``direct_response_intent`` and value containing ``response`` and ``ranking``
:Requires: ``dense_features`` and/or ``sparse_features`` for user message and response

:Output-Example:

    .. code-block:: json

        {
            "response_selector": {
              "faq": {
                "response": {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
                "ranking": [
                    {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"},
                    {"confidence": 0.2134543431, "name": "You can ask me about how to get started"}
                ]
              }
            }
        }

:Description:

    DIET Selector component can be used to build a response retrieval model to directly predict a bot response from
    a set of candidate responses. The prediction of this model is used by :ref:`retrieval-actions`.
    It embeds user inputs and response labels into the same space and follows the exact same
    neural network architecture and optimization as the ``DIETClassifier``.

    .. note:: If during prediction time a message contains **only** words unseen during training,
              and no Out-Of-Vacabulary preprocessor was used,
              empty response ``None`` is predicted with confidence ``0.0``.

:Configuration:

    The algorithm includes all the hyperparameters that ``DIETClassifier`` uses.
    In addition, the component can also be configured to train a response selector for a particular retrieval intent.

        - ``retrieval_intent`` sets the name of the intent for which this response selector model is trained.

    Default values:

    .. code-block:: yaml

        pipeline:
        - name: "ResponseSelector"
            # nn architecture
            # sizes of hidden layers before the embedding layer
            # for input words and intent labels,
            # the number of hidden layers is thus equal to the length of this list
            "hidden_layers_sizes": {"text": [], "label": []}
            # Whether to share the hidden layer weights between input words and labels
            "share_hidden_layers": False
            # number of units in transformer
            "transformer_size": 256
            # number of transformer layers
            "number_of_transformer_layers": 2
            # number of attention heads in transformer
            "number_of_attention_heads": 4
            # max sequence length
            "maximum_sequence_length": 256
            # use a unidirectional or bidirectional encoder
            "unidirectional_encoder": False
            # if true use key relative embeddings in attention
            "use_key_relative_attention": False
            # if true use key relative embeddings in attention
            "use_value_relative_attention": False
            # max position for relative embeddings
            "max_relative_position": None
            # training parameters
            # initial and final batch sizes - batch size will be
            # linearly increased for each epoch
            "batch_size": [64, 256]
            # how to create batches
            "batch_strategy": "balanced"  # string 'sequence' or 'balanced'
            # number of epochs
            "epochs": 300
            # set random seed to any int to get reproducible results
            "random_seed": None
            # optimizer
            "learning_rate": 0.001
            # embedding parameters
            # default dense dimension used if no dense features are present
            "dense_dimension": {"text": 512, "label": 512}
            # dimension size of embedding vectors
            "embedding_dimension": 20
            # the type of the similarity
            "number_of_negative_examples": 20
            # flag if minimize only maximum similarity over incorrect actions
            "similarity_type": "auto"  # string 'auto' or 'cosine' or 'inner'
            # the type of the loss function
            "loss_type": "softmax"  # string 'softmax' or 'margin'
            # number of top intents to normalize scores for softmax loss_type
            # set to 0 to turn off normalization
            "ranking_length": 10
            # how similar the algorithm should try
            # to make embedding vectors for correct labels
            "maximum_positive_similarity": 0.8  # should be 0.0 < ... < 1.0 for 'cosine'
            # maximum negative similarity for incorrect labels
            "maximum_negative_similarity": -0.4  # should be -1.0 < ... < 1.0 for 'cosine'
            # flag: if true, only minimize the maximum similarity for incorrect labels
            "use_maximum_negative_similarity": True
            # scale loss inverse proportionally to confidence of correct prediction
            "scale_loss": True
            # regularization parameters
            # the scale of regularization
            "regularization_constant": 0.002
            # the scale of how critical the algorithm should be of minimizing the
            # maximum similarity between embeddings of different labels
            "negative_margin_scale": 0.8
            # dropout rate for rnn
            "droprate": 0.2
            # dropout rate for attention
            "droprate_attention": 0
            # if true apply dropout to sparse tensors
            "use_sparse_input_dropout": True
            # visualization of accuracy
            # how often to calculate training accuracy
            "evaluate_every_number_of_epochs": 20  # small values may hurt performance
            # how many examples to use for calculation of training accuracy
            "evaluate_on_number_of_examples": 0  # large values may hurt performance
            # if true random tokens of the input message will be masked and the model
            # should predict those tokens
            "use_masked_language_model": False
            # selector config
            # name of the intent for which this response selector is to be trained
            "retrieval_intent": None


Entity Extractors
-----------------

Entity extractors extract entities, such as person names or locations, from the user input.

MitieEntityExtractor
~~~~~~~~~~~~~~~~~~~~

:Short: MITIE entity extraction (using a `MITIE NER trainer <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/ner_trainer.cpp>`_)
:Outputs: ``entities``
:Requires: :ref:`MitieNLP` and ``tokens``
:Output-Example:

    .. code-block:: json

        {
            "entities": [{
                "value": "New York City",
                "start": 20,
                "end": 33,
                "confidence": null,
                "entity": "city",
                "extractor": "MitieEntityExtractor"
            }]
        }

:Description:
    ``MitieEntityExtractor`` uses the MITIE entity extraction to find entities in a message. The underlying classifier
    is using a multi class linear SVM with a sparse linear kernel and custom features.
    The MITIE component does not provide entity confidence values.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "MitieEntityExtractor"

.. _SpacyEntityExtractor:

SpacyEntityExtractor
~~~~~~~~~~~~~~~~~~~~

:Short: spaCy entity extraction
:Outputs: ``entities``
:Requires: :ref:`SpacyNLP`
:Output-Example:

    .. code-block:: json

        {
            "entities": [{
                "value": "New York City",
                "start": 20,
                "end": 33,
                "confidence": null,
                "entity": "city",
                "extractor": "SpacyEntityExtractor"
            }]
        }

:Description:
    Using spaCy this component predicts the entities of a message. spaCy uses a statistical BILOU transition model.
    As of now, this component can only use the spaCy builtin entity extraction models and can not be retrained.
    This extractor does not provide any confidence scores.

:Configuration:
    Configure which dimensions, i.e. entity types, the spaCy component
    should extract. A full list of available dimensions can be found in
    the `spaCy documentation <https://spacy.io/api/annotation#section-named-entities>`_.
    Leaving the dimensions option unspecified will extract all available dimensions.

    .. code-block:: yaml

        pipeline:
        - name: "SpacyEntityExtractor"
          # dimensions to extract
          dimensions: None


EntitySynonymMapper
~~~~~~~~~~~~~~~~~~~

:Short: Maps synonymous entity values to the same value.
:Outputs: Modifies existing entities that previous entity extraction components found.
:Requires: Nothing
:Description:
    If the training data contains defined synonyms, this component will make sure that detected entity values will
    be mapped to the same value. For example, if your training data contains the following examples:

    .. code-block:: json

        [
            {
              "text": "I moved to New York City",
              "intent": "inform_relocation",
              "entities": [{
                "value": "nyc",
                "start": 11,
                "end": 24,
                "entity": "city",
              }]
            },
            {
              "text": "I got a new flat in NYC.",
              "intent": "inform_relocation",
              "entities": [{
                "value": "nyc",
                "start": 20,
                "end": 23,
                "entity": "city",
              }]
            }
        ]

    This component will allow you to map the entities ``New York City`` and ``NYC`` to ``nyc``. The entity
    extraction will return ``nyc`` even though the message contains ``NYC``. When this component changes an
    existing entity, it appends itself to the processor list of this entity.

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "EntitySynonymMapper"

CRFEntityExtractor
~~~~~~~~~~~~~~~~~~

:Short: Conditional random field (CRF) entity extraction
:Outputs: ``entities``
:Requires: ``tokens`` and ``dense_features`` (optional)
:Output-Example:

    .. code-block:: json

        {
            "entities": [{
                "value":"New York City",
                "start": 20,
                "end": 33,
                "entity": "city",
                "confidence": 0.874,
                "extractor": "CRFEntityExtractor"
            }]
        }

:Description:
    This component implements a conditional random fields (CRF) to do named entity recognition.
    CRFs can be thought of as an undirected Markov chain where the time steps are words
    and the states are entity classes. Features of the words (capitalisation, POS tagging,
    etc.) give probabilities to certain entity classes, as are transitions between
    neighbouring entity tags: the most likely set of tags is then calculated and returned.

    .. note::
        If POS features are used (pos or pos2), you need to have ``SpacyTokenizer`` in your pipeline.

    .. note::
        If "pattern" features are used, you need to have ``RegexFeaturizer`` in your pipeline.

    .. warning::
        ``CRFEntityExtractor`` is deprecated and should be replaced by ``DIETClassifier``. See
        `migration guide <https://rasa.com/docs/rasa/migration-guide/#rasa-1-7-to-rasa-1-8>`_ for more details.

:Configuration:
    You need to configure what kind of features the CRF should use.
    The following features are available:

    ===============  =============================================================================
    Feature Name     Description
    ===============  =============================================================================
    low              Checks if the token is lower case.
    upper            Checks if the token is upper case.
    title            Checks if the token starts with an uppercase character and all remaining
                     characters are lowercased.
    digit            Checks if the token contains just digits.
    prefix5          Take the first five characters of the token.
    prefix2          Take the first two characters of the token.
    suffix5          Take the last five characters of the token.
    suffix3          Take the last three characters of the token.
    suffix2          Take the last two characters of the token.
    suffix1          Take the last character of the token.
    pos              Take the Part-of-Speech tag of the token (``SpacyTokenizer`` required).
    pos2             Take the first two characters of the Part-of-Speech tag of the token
                     (``SpacyTokenizer`` required).
    pattern          Take the patterns defined by ``RegexFeaturizer``.
    bias             Add an additional "bias" feature to the list of features.
    ===============  =============================================================================

    As the featurizer is moving over the tokens in a user message with a sliding window, you can define features for
    previous tokens, the current token, and the next tokens in the sliding window.
    You define the features as [before, token, after] array.

    Additional you can set a flag to determine whether to use the BILOU tagging schema or not.

        - ``BILOU_flag`` determines whether to use BILOU tagging or not.

    .. code-block:: yaml

        pipeline:
        - name: "CRFEntityExtractor"
            # BILOU_flag determines whether to use BILOU tagging or not.
            # More rigorous however requires more examples per entity
            # rule of thumb: use only if more than 100 egs. per entity
            "BILOU_flag": True,
            # crf_features is [before, token, after] array with before, token,
            # after holding keys about which features to use for each token,
            # for example, 'title' in array before will have the feature
            # "is the preceding token in title case?"
            # POS features require SpacyTokenizer
            # pattern feature require RegexFeaturizer
            "features": [
                ["low", "title", "upper"],
                [
                    "bias",
                    "low",
                    "prefix5",
                    "prefix2",
                    "suffix5",
                    "suffix3",
                    "suffix2",
                    "upper",
                    "title",
                    "digit",
                    "pattern",
                ],
                ["low", "title", "upper"],
            ],
            # The maximum number of iterations for optimization algorithms.
            "max_iterations": 50,
            # weight of the L1 regularization
            "L1_c": 0.1,
            # weight of the L2 regularization
            "L2_c": 0.1,

.. _DucklingHTTPExtractor:

DucklingHTTPExtractor
~~~~~~~~~~~~~~~~~~~~~

:Short: Duckling lets you extract common entities like dates,
        amounts of money, distances, and others in a number of languages.
:Outputs: ``entities``
:Requires: Nothing
:Output-Example:

    .. code-block:: json

        {
            "entities": [{
                "end": 53,
                "entity": "time",
                "start": 48,
                "value": "2017-04-10T00:00:00.000+02:00",
                "confidence": 1.0,
                "extractor": "DucklingHTTPExtractor"
            }]
        }

:Description:
    To use this component you need to run a duckling server. The easiest
    option is to spin up a docker container using
    ``docker run -p 8000:8000 rasa/duckling``.

    Alternatively, you can `install duckling directly on your
    machine <https://github.com/facebook/duckling#quickstart>`_ and start the server.

    Duckling allows to recognize dates, numbers, distances and other structured entities
    and normalizes them.
    Please be aware that duckling tries to extract as many entity types as possible without
    providing a ranking. For example, if you specify both ``number`` and ``time`` as dimensions
    for the duckling component, the component will extract two entities: ``10`` as a number and
    ``in 10 minutes`` as a time from the text ``I will be there in 10 minutes``. In such a
    situation, your application would have to decide which entity type is be the correct one.
    The extractor will always return `1.0` as a confidence, as it is a rule
    based system.

:Configuration:
    Configure which dimensions, i.e. entity types, the duckling component
    should extract. A full list of available dimensions can be found in
    the `duckling documentation <https://duckling.wit.ai/>`_.
    Leaving the dimensions option unspecified will extract all available dimensions.

    .. code-block:: yaml

        pipeline:
        - name: "DucklingHTTPExtractor"
          # url of the running duckling server
          url: "http://localhost:8000"
          # dimensions to extract
          dimensions: ["time", "number", "amount-of-money", "distance"]
          # allows you to configure the locale, by default the language is
          # used
          locale: "de_DE"
          # if not set the default timezone of Duckling is going to be used
          # needed to calculate dates from relative expressions like "tomorrow"
          timezone: "Europe/Berlin"
          # Timeout for receiving response from http url of the running duckling server
          # if not set the default timeout of duckling http url is set to 3 seconds.
          timeout : 3


Combined Entity Extractors and Intent Classifiers
-------------------------------------------------

.. _diet-classifier:

DIETClassifier
~~~~~~~~~~~~~~

:Short: Dual Intent Entity Transformer (DIET) used for intent classification and entity extraction
:Outputs: ``entities``, ``intent`` and ``intent_ranking``
:Requires: ``dense_features`` and/or ``sparse_features`` for user message and intent (optional)
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.8343},
            "intent_ranking": [
                {
                    "confidence": 0.385910906220309,
                    "name": "goodbye"
                },
                {
                    "confidence": 0.28161531595656784,
                    "name": "restaurant_search"
                }
            ],
            "entities": [{
                "end": 53,
                "entity": "time",
                "start": 48,
                "value": "2017-04-10T00:00:00.000+02:00",
                "confidence": 1.0,
                "extractor": "DIETClassifier"
            }]
        }

:Description:
    DIET (Dual Intent and Entity Transformer) is a multi-task architecture for intent classification and entity
    recognition. The architecture is based on a transformer which is shared for both tasks.
    A sequence of entity labels is predicted through a Conditional Random Field (CRF) tagging layer on top of the
    transformer output sequence corresponding to the input sequence of tokens.
    The transformer output for the ``__CLS__`` token and intent labels are embedded into a single semantic vector
    space. We use the dot-product loss to maximize the similarity with the target label and minimize
    similarities with negative samples.

    .. note:: If during prediction time a message contains **only** words unseen during training
              and no Out-Of-Vacabulary preprocessor was used,
              an empty intent ``None`` is predicted with confidence ``0.0``.

:Configuration:

    The following hyperparameters can be set:

        - neural network's architecture:

            - ``hidden_layers_sizes.text`` sets a list of hidden layer sizes before
              the embedding layer for user inputs, the number of hidden layers
              is equal to the length of the list.
            - ``hidden_layers_sizes.label`` sets a list of hidden layer sizes before
              the embedding layer for intent labels, the number of hidden layers
              is equal to the length of the list.
            - ``share_hidden_layers`` if set to True, shares the hidden layers between user inputs and intent label.
            - ``transformer_size`` sets the size of the transformer.
            - ``number_of_transformer_layers`` sets the number of transformer layers to use.
            - ``number_of_attention_heads`` sets the number of attention heads to use.
            - ``maximum_sequence_length`` sets the maximum length of sequence.
            - ``unidirectional_encoder`` specifies whether to use a unidirectional or bidirectional encoder.
            - ``use_key_relative_attention`` if true use key relative embeddings in attention.
            - ``use_value_relative_attention`` if true use key relative embeddings in attention.
            - ``max_relative_position`` sets the max position for relative embeddings.

        - training:

            - ``batch_size`` sets the number of training examples in one
              forward/backward pass, the higher the batch size, the more
              memory space you'll need.
            - ``batch_strategy`` sets the type of batching strategy,
              it should be either ``sequence`` or ``balanced``.
            - ``epochs`` sets the number of times the algorithm will see
              training data, where one ``epoch`` equals one forward pass and
              one backward pass of all the training examples.
            - ``random_seed`` if set you will get reproducible
              training results for the same inputs.
            - ``learning_rate`` sets the initial learning rate of the optimizer.

        - embedding:

            - ``dense_dimension.text`` sets the dense dimensions for user inputs to use for sparse
              tensors if no dense features are present.
            - ``dense_dimension.label`` sets the dense dimensions for intent labels to use for sparse
              tensors if no dense features are present.
            - ``embedding_dimension`` sets the dimension of embedding space.
            - ``number_of_negative_examples`` sets the number of incorrect intent labels.
              The algorithm will minimize their similarity to the user
              input during training.
            - ``similarity_type`` sets the type of the similarity,
              it should be either ``auto``, ``cosine`` or ``inner``,
              if ``auto``, it will be set depending on ``loss_type``,
              ``inner`` for ``softmax``, ``cosine`` for ``margin``.
            - ``loss_type`` sets the type of the loss function,
              it should be either ``softmax`` or ``margin``.
            - ``ranking_length`` defines the number of top confidences over
              which to normalize ranking results if ``loss_type: "softmax"``.
              To turn off normalization set it to 0.
            - ``maximum_positive_similarity`` controls how similar the algorithm should try
              to make embedding vectors for correct intent labels,
              used only if ``loss_type`` is set to ``margin``.
            - ``maximum_negative_similarity`` controls maximum negative similarity for
              incorrect intents, used only if ``loss_type`` is set to ``margin``.
            - ``use_maximum_negative_similarity`` if ``true`` the algorithm only
              minimizes maximum similarity over incorrect intent labels,
              used only if ``loss_type`` is set to ``margin``.
            - ``scale_loss`` if ``true`` the algorithm will downscale the loss
              for examples where correct label is predicted with high confidence,
              used only if ``loss_type`` is set to ``softmax``.

        - regularization:

            - ``regularization_constant`` sets the scale of L2 regularization.
            - ``negative_margin_scale`` sets the scale of how important is to minimize
              the maximum similarity between embeddings of different intent labels.
            - ``droprate`` sets the dropout rate, it should be
              between ``0`` and ``1``, e.g. ``droprate=0.1`` would drop out ``10%`` of input units.
            - ``droprate_attention`` sets the dropout rate for attention, it should be
              between ``0`` and ``1``, e.g. ``droprate_attention=0.1`` would drop out ``10%`` of input units.
            - ``use_sparse_input_dropout`` specifies whether to apply dropout to sparse tensors or not.

        - model configuration:

            - ``use_masked_language_model`` specifies whether to apply masking or not.
            - ``intent_classification`` indicates whether intent classification should be performed or not.
            - ``entity_recognition`` indicates whether entity recognition should be performed or not.
            - ``BILOU_flag`` determines whether to use BILOU tagging or not.

    .. note:: For ``cosine`` similarity ``maximum_positive_similarity`` and ``maximum_negative_similarity`` should
              be between ``-1`` and ``1``.

    .. note:: There is an option to use linearly increasing batch size. The idea comes from
              `<https://arxiv.org/abs/1711.00489>`_.
              In order to do it pass a list to ``batch_size``, e.g. ``"batch_size": [64, 256]`` (default behaviour).
              If constant ``batch_size`` is required, pass an ``int``, e.g. ``"batch_size": 64``.

    .. note:: Parameter ``maximum_negative_similarity`` is set to a negative value to mimic the original
              starspace algorithm in the case ``maximum_negative_similarity = maximum_positive_similarity``
              and ``use_maximum_negative_similarity = False``.
              See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.

    Default values:

    .. code-block:: yaml

        pipeline:
        - name: "DIETClassifier"
            # nn architecture
            # sizes of hidden layers before the embedding layer
            # for input words and intent labels,
            # the number of hidden layers is thus equal to the length of this list
            "hidden_layers_sizes": {"text": [], "label": []}
            # Whether to share the hidden layer weights between input words and labels
            "share_hidden_layers": False
            # number of units in transformer
            "transformer_size": 256
            # number of transformer layers
            "number_of_transformer_layers": 2
            # number of attention heads in transformer
            "number_of_attention_heads": 4
            # max sequence length
            "maximum_sequence_length": 256
            # use a unidirectional or bidirectional encoder
            "unidirectional_encoder": False
            # if true use key relative embeddings in attention
            "use_key_relative_attention": False
            # if true use key relative embeddings in attention
            "use_value_relative_attention": False
            # max position for relative embeddings
            "max_relative_position": None
            # training parameters
            # initial and final batch sizes - batch size will be
            # linearly increased for each epoch
            "batch_size": [64, 256]
            # how to create batches
            "batch_strategy": "balanced"  # string 'sequence' or 'balanced'
            # number of epochs
            "epochs": 300
            # set random seed to any int to get reproducible results
            "random_seed": None
            # optimizer
            "learning_rate": 0.001
            # embedding parameters
            # default dense dimension used if no dense features are present
            "dense_dimension": {"text": 512, "label": 20}
            # dimension size of embedding vectors
            "embedding_dimension": 20
            # the type of the similarity
            "number_of_negative_examples": 20
            # flag if minimize only maximum similarity over incorrect actions
            "similarity_type": "auto"  # string 'auto' or 'cosine' or 'inner'
            # the type of the loss function
            "loss_type": "softmax"  # string 'softmax' or 'margin'
            # number of top intents to normalize scores for softmax loss_type
            # set to 0 to turn off normalization
            "ranking_length": 10
            # how similar the algorithm should try
            # to make embedding vectors for correct labels
            "maximum_positive_similarity": 0.8  # should be 0.0 < ... < 1.0 for 'cosine'
            # maximum negative similarity for incorrect labels
            "maximum_negative_similarity": -0.4  # should be -1.0 < ... < 1.0 for 'cosine'
            # flag: if true, only minimize the maximum similarity for incorrect labels
            "use_maximum_negative_similarity": True
            # scale loss inverse proportionally to confidence of correct prediction
            "scale_loss": True
            # regularization parameters
            # the scale of regularization
            "regularization_constant": 0.002
            # the scale of how critical the algorithm should be of minimizing the
            # maximum similarity between embeddings of different labels
            "negative_margin_scale": 0.8
            # dropout rate for rnn
            "droprate": 0.2
            # dropout rate for attention
            "droprate_attention": 0
            # if true apply dropout to sparse tensors
            "use_sparse_input_dropout": True
            # visualization of accuracy
            # how often to calculate training accuracy
            "evaluate_every_number_of_epochs": 20  # small values may hurt performance
            # how many examples to use for calculation of training accuracy
            "evaluate_on_number_of_examples": 0  # large values may hurt performance
            # model config
            # if true intent classification is trained and intent predicted
            "intent_classification": True
            # if true named entity recognition is trained and entities predicted
            "entity_recognition": True
            # if true random tokens of the input message will be masked and the model
            # should predict those tokens
            "use_masked_language_model": False
            # BILOU_flag determines whether to use BILOU tagging or not.
            # More rigorous however requires more examples per entity
            # rule of thumb: use only if more than 100 egs. per entity
            "BILOU_flag": True