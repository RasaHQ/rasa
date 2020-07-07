:desc: Customize the components and parameters of Rasa's Machine Learning based
       Natural Language Understanding pipeline

.. _components:

Components
==========

.. edit-link::

This is a reference of the configuration options for every built-in
component in Rasa Open Source. If you want to build a custom component, check
out :ref:`custom-nlu-components`.

.. contents::
   :local:


Word Vector Sources
-------------------

The following components load pre-trained models that are needed if you want to use pre-trained
word vectors in your pipeline.

.. _MitieNLP:

MitieNLP
~~~~~~~~

:Short: MITIE initializer
:Outputs: Nothing
:Requires: Nothing
:Description:
    Initializes MITIE structures. Every MITIE component relies on this,
    hence this should be put at the beginning
    of every pipeline that uses any MITIE components.
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
    You need to specify the language model to use.
    By default the language configured in the pipeline will be used as the language model name.
    If the spaCy model to be used has a name that is different from the language tag (``"en"``, ``"de"``, etc.),
    the model name can be specified using the configuration variable ``model``.
    The name will be passed to ``spacy.load(name)``.

    .. code-block:: yaml

        pipeline:
        - name: "SpacyNLP"
          # language model to load
          model: "en_core_web_md"

          # when retrieving word vectors, this will decide if the casing
          # of the word is relevant. E.g. `hello` and `Hello` will
          # retrieve the same vector, if set to `False`. For some
          # applications and models it makes sense to differentiate
          # between these two words, therefore setting this to `True`.
          case_sensitive: False

    For more information on how to download the spaCy models, head over to
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

     .. note:: To use ``HFTransformersNLP`` component, install Rasa Open Source with ``pip install rasa[transformers]``.

:Configuration:
    You should specify what language model to load via the parameter ``model_name``. See the below table for the
    available language models.
    Additionally, you can also specify the architecture variation of the chosen language model by specifying the
    parameter ``model_weights``.
    The full list of supported architectures can be found
    `here <https://huggingface.co/transformers/pretrained_models.html>`__.
    If left empty, it uses the default model architecture that original Transformers library loads (see table below).

    .. code-block:: none

        +----------------+--------------+-------------------------+
        | Language Model | Parameter    | Default value for       |
        |                | "model_name" | "model_weights"         |
        +----------------+--------------+-------------------------+
        | BERT           | bert         | bert-base-uncased       |
        +----------------+--------------+-------------------------+
        | GPT            | gpt          | openai-gpt              |
        +----------------+--------------+-------------------------+
        | GPT-2          | gpt2         | gpt2                    |
        +----------------+--------------+-------------------------+
        | XLNet          | xlnet        | xlnet-base-cased        |
        +----------------+--------------+-------------------------+
        | DistilBERT     | distilbert   | distilbert-base-uncased |
        +----------------+--------------+-------------------------+
        | RoBERTa        | roberta      | roberta-base            |
        +----------------+--------------+-------------------------+

    The following configuration loads the language model BERT:

    .. code-block:: yaml

        pipeline:
          - name: HFTransformersNLP
            # Name of the language model to use
            model_name: "bert"
            # Pre-Trained weights to be loaded
            model_weights: "bert-base-uncased"
            
            # An optional path to a specific directory to download and cache the pre-trained model weights.
            # The `default` cache_dir is the same as https://huggingface.co/transformers/serialization.html#cache-directory .
            cache_dir: null

.. _tokenizers:

Tokenizers
----------

Tokenizers split text into tokens.
If you want to split intents into multiple labels, e.g. for predicting multiple intents or for
modeling hierarchical intent structure, use the following flags with any tokenizer:

- ``intent_tokenization_flag`` indicates whether to tokenize intent labels or not. Set it to ``True``, so that intent
  labels are tokenized.
- ``intent_split_symbol`` sets the delimiter string to split the intent labels, default is underscore
  (``_``).


.. _WhitespaceTokenizer:

WhitespaceTokenizer
~~~~~~~~~~~~~~~~~~~

:Short: Tokenizer using whitespaces as a separator
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified)
:Requires: Nothing
:Description:
    Creates a token for every whitespace separated character sequence.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "WhitespaceTokenizer"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"
          # Regular expression to detect tokens
          "token_pattern": None


JiebaTokenizer
~~~~~~~~~~~~~~

:Short: Tokenizer using Jieba for Chinese language
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified)
:Requires: Nothing
:Description:
    Creates tokens using the Jieba tokenizer specifically for Chinese
    language. It will only work for the Chinese language.

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
          # Regular expression to detect tokens
          "token_pattern": None


MitieTokenizer
~~~~~~~~~~~~~~

:Short: Tokenizer using MITIE
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified)
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
          # Regular expression to detect tokens
          "token_pattern": None

SpacyTokenizer
~~~~~~~~~~~~~~

:Short: Tokenizer using spaCy
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified)
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
          # Regular expression to detect tokens
          "token_pattern": None

.. _ConveRTTokenizer:

ConveRTTokenizer
~~~~~~~~~~~~~~~~

:Short: Tokenizer using `ConveRT <https://github.com/PolyAI-LDN/polyai-models#convert>`__ model.
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified)
:Requires: Nothing
:Description:
    Creates tokens using the ConveRT tokenizer. Must be used whenever the :ref:`ConveRTFeaturizer` is used.

    .. note::
        Since ``ConveRT`` model is trained only on an English corpus of conversations, this tokenizer should only
        be used if your training data is in English language.

    .. note::
        To use ``ConveRTTokenizer``, install Rasa Open Source with ``pip install rasa[convert]``.


:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "ConveRTTokenizer"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"
          # Regular expression to detect tokens
          "token_pattern": None

.. _LanguageModelTokenizer:

LanguageModelTokenizer
~~~~~~~~~~~~~~~~~~~~~~

:Short: Tokenizer from pre-trained language models
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified)
:Requires: :ref:`HFTransformersNLP`
:Description:
    Creates tokens using the pre-trained language model specified in upstream :ref:`HFTransformersNLP` component.
    Must be used whenever the :ref:`LanguageModelFeaturizer` is used.
:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "LanguageModelTokenizer"
          # Flag to check whether to split intents
          "intent_tokenization_flag": False
          # Symbol on which intent should be split
          "intent_split_symbol": "_"


.. _text-featurizers:

Text Featurizers
----------------

Text featurizers are divided into two different categories: sparse featurizers and dense featurizers.
Sparse featurizers are featurizers that return feature vectors with a lot of missing values, e.g. zeros.
As those feature vectors would normally take up a lot of memory, we store them as sparse features.
Sparse features only store the values that are non zero and their positions in the vector.
Thus, we save a lot of memory and are able to train on larger datasets.

All featurizers can return two different kind of features: sequence features and sentence features.
The sequence features are a matrix of size ``(number-of-tokens x feature-dimension)``.
The matrix contains a feature vector for every token in the sequence.
This allows us to train sequence models.
The sentence features are represented by a matrix of size ``(1 x feature-dimension)``.
It contains the feature vector for the complete utterance.
The sentence features can be used in any bag-of-words model.
The corresponding classifier can therefore decide what kind of features to use.
Note: The ``feature-dimension`` for sequence and sentence features does not have to be the same.

.. _MitieFeaturizer:

MitieFeaturizer
~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using the MITIE featurizer.
:Outputs: ``dense_features`` for user messages and responses
:Requires: :ref:`MitieNLP`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response classification using the MITIE
    featurizer.

    .. note::

        NOT used by the ``MitieIntentClassifier`` component. But can be used by any component later in the pipeline
        that makes use of ``dense_features``.

:Configuration:
    The sentence vector, i.e. the vector of the complete utterance, can be calculated in two different ways, either via
    mean or via max pooling. You can specify the pooling method in your configuration file with the option ``pooling``.
    The default pooling method is set to ``mean``.

    .. code-block:: yaml

        pipeline:
        - name: "MitieFeaturizer"
          # Specify what pooling operation should be used to calculate the vector of
          # the complete utterance. Available options: 'mean' and 'max'.
          "pooling": "mean"


.. _SpacyFeaturizer:

SpacyFeaturizer
~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using the spaCy featurizer.
:Outputs: ``dense_features`` for user messages and responses
:Requires: :ref:`SpacyNLP`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response classification using the spaCy
    featurizer.
:Configuration:
    The sentence vector, i.e. the vector of the complete utterance, can be calculated in two different ways, either via
    mean or via max pooling. You can specify the pooling method in your configuration file with the option ``pooling``.
    The default pooling method is set to ``mean``.

    .. code-block:: yaml

        pipeline:
        - name: "SpacyFeaturizer"
          # Specify what pooling operation should be used to calculate the vector of
          # the complete utterance. Available options: 'mean' and 'max'.
          "pooling": "mean"


.. _ConveRTFeaturizer:

ConveRTFeaturizer
~~~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using
    `ConveRT <https://github.com/PolyAI-LDN/polyai-models>`__ model.
:Outputs: ``dense_features`` for user messages and responses
:Requires: :ref:`ConveRTTokenizer`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response selection.
    It uses the `default signature <https://github.com/PolyAI-LDN/polyai-models#tfhub-signatures>`_ to compute vector
    representations of input text.

    .. note::
        Since ``ConveRT`` model is trained only on an English corpus of conversations, this featurizer should only
        be used if your training data is in English language.

    .. note::
        To use ``ConveRTTokenizer``, install Rasa Open Source with ``pip install rasa[convert]``.

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "ConveRTFeaturizer"


.. _LanguageModelFeaturizer:

LanguageModelFeaturizer
~~~~~~~~~~~~~~~~~~~~~~~

:Short:
    Creates a vector representation of user message and response (if specified) using a pre-trained language model.
:Outputs: ``dense_features`` for user messages and responses
:Requires: :ref:`HFTransformersNLP` and :ref:`LanguageModelTokenizer`
:Type: Dense featurizer
:Description:
    Creates features for entity extraction, intent classification, and response selection.
    Uses the pre-trained language model specified in upstream :ref:`HFTransformersNLP` component to compute vector
    representations of input text.

    .. note::
        Please make sure that you use a language model which is pre-trained on the same language corpus as that of your
        training data.

:Configuration:

    Include :ref:`HFTransformersNLP` and :ref:`LanguageModelTokenizer` components before this component. Use
    :ref:`LanguageModelTokenizer` to ensure tokens are correctly set for all components throughout the pipeline.

    .. code-block:: yaml

        pipeline:
        - name: "LanguageModelFeaturizer"


.. _RegexFeaturizer:

RegexFeaturizer
~~~~~~~~~~~~~~~

:Short: Creates a vector representation of user message using regular expressions.
:Outputs: ``sparse_features`` for user messages and ``tokens.pattern``
:Requires: ``tokens``
:Type: Sparse featurizer
:Description:
    Creates features for entity extraction and intent classification.
    During training the ``RegexFeaturizer`` creates a list of regular expressions defined in the training
    data format.
    For each regex, a feature will be set marking whether this expression was found in the user message or not.
    All features will later be fed into an intent classifier / entity extractor to simplify classification (assuming
    the classifier has learned during the training phase, that this set feature indicates a certain intent / entity).
    Regex features for entity extraction are currently only supported by the :ref:`CRFEntityExtractor` and the
    :ref:`diet-classifier` components!

:Configuration:
    Make the featurizer case insensitive by adding the ``case_sensitive: False`` option, the default being
    ``case_sensitive: True``.

    .. code-block:: yaml

        pipeline:
        - name: "RegexFeaturizer"
          # Text will be processed with case sensitive as default
          "case_sensitive": True

.. _CountVectorsFeaturizer:

CountVectorsFeaturizer
~~~~~~~~~~~~~~~~~~~~~~

:Short: Creates bag-of-words representation of user messages, intents, and responses.
:Outputs: ``sparse_features`` for user messages, intents, and responses
:Requires: ``tokens``
:Type: Sparse featurizer
:Description:
    Creates features for intent classification and response selection.
    Creates bag-of-words representation of user message, intent, and response using
    `sklearn's CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_.
    All tokens which consist only of digits (e.g. 123 and 99 but not a123d) will be assigned to the same feature.

:Configuration:
    See `sklearn's CountVectorizer docs <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_
    for detailed description of the configuration parameters.

    This featurizer can be configured to use word or character n-grams, using the ``analyzer`` configuration parameter.
    By default ``analyzer`` is set to ``word`` so word token counts are used as features.
    If you want to use character n-grams, set ``analyzer`` to ``char`` or ``char_wb``.
    The lower and upper boundaries of the n-grams can be configured via the parameters ``min_ngram`` and ``max_ngram``.
    By default both of them are set to ``1``.

    .. note::
        Option ``char_wb`` creates character n-grams only from text inside word boundaries;
        n-grams at the edges of words are padded with space.
        This option can be used to create `Subword Semantic Hashing <https://arxiv.org/abs/1810.07150>`_.

    .. note::
        For character n-grams do not forget to increase ``min_ngram`` and ``max_ngram`` parameters.
        Otherwise the vocabulary will contain only single letters.

    Handling Out-Of-Vocabulary (OOV) words:

        .. note:: Enabled only if ``analyzer`` is ``word``.

        Since the training is performed on limited vocabulary data, it cannot be guaranteed that during prediction
        an algorithm will not encounter an unknown word (a word that were not seen during training).
        In order to teach an algorithm how to treat unknown words, some words in training data can be substituted
        by generic word ``OOV_token``.
        In this case during prediction all unknown words will be treated as this generic word ``OOV_token``.

        For example, one might create separate intent ``outofscope`` in the training data containing messages of
        different number of ``OOV_token`` s and maybe some additional general words.
        Then an algorithm will likely classify a message with unknown words as this intent ``outofscope``.

        You can either set the ``OOV_token`` or a list of words ``OOV_words``:

            - ``OOV_token`` set a keyword for unseen words; if training data contains ``OOV_token`` as words in some
              messages, during prediction the words that were not seen during training will be substituted with
              provided ``OOV_token``; if ``OOV_token=None`` (default behavior) words that were not seen during
              training will be ignored during prediction time;
            - ``OOV_words`` set a list of words to be treated as ``OOV_token`` during training; if a list of words
              that should be treated as Out-Of-Vocabulary is known, it can be set to ``OOV_words`` instead of manually
              changing it in training data or using custom preprocessor.

        .. note::
            This featurizer creates a bag-of-words representation by **counting** words,
            so the number of ``OOV_token`` in the sentence might be important.

        .. note::
            Providing ``OOV_words`` is optional, training data can contain ``OOV_token`` input manually or by custom
            additional preprocessor.
            Unseen words will be substituted with ``OOV_token`` **only** if this token is present in the training
            data or ``OOV_words`` list is provided.

    If you want to share the vocabulary between user messages and intents, you need to set the option
    ``use_shared_vocab`` to ``True``. In that case a common vocabulary set between tokens in intents and user messages
    is build.

    .. code-block:: yaml

        pipeline:
        - name: "CountVectorsFeaturizer"
          # Analyzer to use, either 'word', 'char', or 'char_wb'
          "analyzer": "word"
          # Set the lower and upper boundaries for the n-grams
          "min_ngram": 1
          "max_ngram": 1
          # Set the out-of-vocabulary token
          "OOV_token": "_oov_"
          # Whether to use a shared vocab
          "use_shared_vocab": False

    .. container:: toggle

        .. container:: header

            The above configuration parameters are the ones you should configure to fit your model to your data.
            However, additional parameters exist that can be adapted.

        .. code-block:: none

         +-------------------+-------------------------+--------------------------------------------------------------+
         | Parameter         | Default Value           | Description                                                  |
         +===================+=========================+==============================================================+
         | use_shared_vocab  | False                   | If set to 'True' a common vocabulary is used for labels      |
         |                   |                         | and user message.                                            |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | analyzer          | word                    | Whether the features should be made of word n-gram or        |
         |                   |                         | character n-grams. Option ‘char_wb’ creates character        |
         |                   |                         | n-grams only from text inside word boundaries;               |
         |                   |                         | n-grams at the edges of words are padded with space.         |
         |                   |                         | Valid values: 'word', 'char', 'char_wb'.                     |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | strip_accents     | None                    | Remove accents during the pre-processing step.               |
         |                   |                         | Valid values: 'ascii', 'unicode', 'None'.                    |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | stop_words        | None                    | A list of stop words to use.                                 |
         |                   |                         | Valid values: 'english' (uses an internal list of            |
         |                   |                         | English stop words), a list of custom stop words, or         |
         |                   |                         | 'None'.                                                      |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | min_df            | 1                       | When building the vocabulary ignore terms that have a        |
         |                   |                         | document frequency strictly lower than the given threshold.  |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | max_df            | 1                       | When building the vocabulary ignore terms that have a        |
         |                   |                         | document frequency strictly higher than the given threshold  |
         |                   |                         | (corpus-specific stop words).                                |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | min_ngram         | 1                       | The lower boundary of the range of n-values for different    |
         |                   |                         | word n-grams or char n-grams to be extracted.                |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | max_ngram         | 1                       | The upper boundary of the range of n-values for different    |
         |                   |                         | word n-grams or char n-grams to be extracted.                |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | max_features      | None                    | If not 'None', build a vocabulary that only consider the top |
         |                   |                         | max_features ordered by term frequency across the corpus.    |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | lowercase         | True                    | Convert all characters to lowercase before tokenizing.       |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | OOV_token         | None                    | Keyword for unseen words.                                    |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | OOV_words         | []                      | List of words to be treated as 'OOV_token' during training.  |
         +-------------------+-------------------------+--------------------------------------------------------------+
         | alias             | CountVectorFeaturizer   | Alias name of featurizer.                                    |
         +-------------------+-------------------------+--------------------------------------------------------------+


.. _LexicalSyntacticFeaturizer:

LexicalSyntacticFeaturizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

:Short: Creates lexical and syntactic features for a user message to support entity extraction.
:Outputs: ``sparse_features`` for user messages
:Requires: ``tokens``
:Type: Sparse featurizer
:Description:
    Creates features for entity extraction.
    Moves with a sliding window over every token in the user message and creates features according to the
    configuration (see below). As a default configuration is present, you don't need to specify a configuration.
:Configuration:
    You can configure what kind of lexical and syntactic features the featurizer should extract.
    The following features are available:

    .. code-block:: none

        ==============  ==========================================================================================
        Feature Name    Description
        ==============  ==========================================================================================
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
        pos             Take the Part-of-Speech tag of the token (``SpacyTokenizer`` required).
        pos2            Take the first two characters of the Part-of-Speech tag of the token
                        (``SpacyTokenizer`` required).
        ==============  ==========================================================================================

    As the featurizer is moving over the tokens in a user message with a sliding window, you can define features for
    previous tokens, the current token, and the next tokens in the sliding window.
    You define the features as a [before, token, after] array.
    If you want to define features for the token before, the current token, and the token after,
    your features configuration would look like this:

    .. code-block:: yaml

        pipeline:
        - name: LexicalSyntacticFeaturizer
          "features": [
            ["low", "title", "upper"],
            ["BOS", "EOS", "low", "upper", "title", "digit"],
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
:Requires: ``tokens`` for user message and :ref:`MitieNLP`
:Output-Example:

    .. code-block:: json

        {
            "intent": {"name": "greet", "confidence": 0.98343}
        }

:Description:
    This classifier uses MITIE to perform intent classification. The underlying classifier
    is using a multi-class linear SVM with a sparse linear kernel (see
    `MITIE trainer code <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/text_categorizer_trainer.cpp#L222>`_).

    .. note:: This classifier does not rely on any featurizer as it extracts features on its own.

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "MitieIntentClassifier"

SklearnIntentClassifier
~~~~~~~~~~~~~~~~~~~~~~~

:Short: Sklearn intent classifier
:Outputs: ``intent`` and ``intent_ranking``
:Requires: ``dense_features`` for user messages
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
    For more information about the algorithm itself, take a look at the
    `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`__
    documentation.

:Configuration:
    During the training of the SVM a hyperparameter search is run to find the best parameter set.
    In the configuration you can specify the parameters that will get tried.

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
          # Gamma parameter of the C-SVM.
          "gamma": [0.1]
          # We try to find a good number of cross folds to use during
          # intent training, this specifies the max number of folds.
          "max_cross_validation_folds": 5
          # Scoring function used for evaluating the hyper parameters.
          # This can be a name or a function.
          "scoring_function": "f1_weighted"

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
              you have few NLU training data, you can take a look at the recommended pipelines in
              :ref:`choosing-a-pipeline`.

:Configuration:

    .. code-block:: yaml

        pipeline:
        - name: "KeywordIntentClassifier"
          case_sensitive: True


DIETClassifier
~~~~~~~~~~~~~~

:Short: Dual Intent Entity Transformer (DIET) used for intent classification and entity extraction
:Description:
    You can find the detailed description of the :ref:`diet-classifier` under the section
    `Combined Entity Extractors and Intent Classifiers`.

Entity Extractors
-----------------

Entity extractors extract entities, such as person names or locations, from the user message.

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

    .. note:: This entity extractor does not rely on any featurizer as it extracts features on its own.

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
          dimensions: ["PERSON", "LOC", "ORG", "PRODUCT"]


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

.. _CRFEntityExtractor:

CRFEntityExtractor
~~~~~~~~~~~~~~~~~~

:Short: Conditional random field (CRF) entity extraction
:Outputs: ``entities``
:Requires: ``tokens`` and ``dense_features`` (optional)
:Output-Example:

    .. code-block:: json

        {
            "entities": [{
                "value": "New York City",
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
    and the states are entity classes. Features of the words (capitalization, POS tagging,
    etc.) give probabilities to certain entity classes, as are transitions between
    neighbouring entity tags: the most likely set of tags is then calculated and returned.

:Configuration:
    ``CRFEntityExtractor`` has a list of default features to use.
    However, you can overwrite the default configuration.
    The following features are available:

    .. code-block:: none

        ==============  ==========================================================================================
        Feature Name    Description
        ==============  ==========================================================================================
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
        pos             Take the Part-of-Speech tag of the token (``SpacyTokenizer`` required).
        pos2            Take the first two characters of the Part-of-Speech tag of the token
                        (``SpacyTokenizer`` required).
        pattern         Take the patterns defined by ``RegexFeaturizer``.
        bias            Add an additional "bias" feature to the list of features.
        ==============  ==========================================================================================

    As the featurizer is moving over the tokens in a user message with a sliding window, you can define features for
    previous tokens, the current token, and the next tokens in the sliding window.
    You define the features as [before, token, after] array.

    Additional you can set a flag to determine whether to use the BILOU tagging schema or not.

        - ``BILOU_flag`` determines whether to use BILOU tagging or not. Default ``True``.

    .. code-block:: yaml

        pipeline:
        - name: "CRFEntityExtractor"
          # BILOU_flag determines whether to use BILOU tagging or not.
          "BILOU_flag": True
          # features to extract in the sliding window
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
          ]
          # The maximum number of iterations for optimization algorithms.
          "max_iterations": 50
          # weight of the L1 regularization
          "L1_c": 0.1
          # weight of the L2 regularization
          "L2_c": 0.1
          # Name of dense featurizers to use.
          # If list is empty all available dense features are used.
          "featurizers": []

    .. note::
        If POS features are used (``pos`` or ``pos2`), you need to have ``SpacyTokenizer`` in your pipeline.

    .. note::
        If "``pattern` features are used, you need to have ``RegexFeaturizer`` in your pipeline.

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

DIETClassifier
~~~~~~~~~~~~~~

:Short: Dual Intent Entity Transformer (DIET) used for intent classification and entity extraction
:Description:
    You can find the detailed description of the :ref:`diet-classifier` under the section
    `Combined Entity Extractors and Intent Classifiers`.


Selectors
----------

Selectors predict a bot response from a set of candidate responses.

.. _response-selector:

ResponseSelector
~~~~~~~~~~~~~~~~

:Short: Response Selector
:Outputs: A dictionary with key as ``direct_response_intent`` and value containing ``response`` and ``ranking``
:Requires: ``dense_features`` and/or ``sparse_features`` for user messages and response

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
    neural network architecture and optimization as the :ref:`diet-classifier`.

    .. note:: If during prediction time a message contains **only** words unseen during training
              and no Out-Of-Vocabulary preprocessor was used, an empty response ``None`` is predicted with confidence
              ``0.0``. This might happen if you only use the :ref:`CountVectorsFeaturizer` with a ``word`` analyzer
              as featurizer. If you use the ``char_wb`` analyzer, you should always get a response with a confidence
              value ``> 0.0``.

:Configuration:

    The algorithm includes almost all the hyperparameters that :ref:`diet-classifier` uses.
    If you want to adapt your model, start by modifying the following parameters:

        - ``epochs``:
          This parameter sets the number of times the algorithm will see the training data (default: ``300``).
          One ``epoch`` is equals to one forward pass and one backward pass of all the training examples.
          Sometimes the model needs more epochs to properly learn.
          Sometimes more epochs don't influence the performance.
          The lower the number of epochs the faster the model is trained.
        - ``hidden_layers_sizes``:
          This parameter allows you to define the number of feed forward layers and their output
          dimensions for user messages and intents (default: ``text: [256, 128], label: [256, 128]``).
          Every entry in the list corresponds to a feed forward layer.
          For example, if you set ``text: [256, 128]``, we will add two feed forward layers in front of
          the transformer. The vectors of the input tokens (coming from the user message) will be passed on to those
          layers. The first layer will have an output dimension of 256 and the second layer will have an output
          dimension of 128. If an empty list is used (default behavior), no feed forward layer will be
          added.
          Make sure to use only positive integer values. Usually, numbers of power of two are used.
          Also, it is usual practice to have decreasing values in the list: next value is smaller or equal to the
          value before.
        - ``embedding_dimension``:
          This parameter defines the output dimension of the embedding layers used inside the model (default: ``20``).
          We are using multiple embeddings layers inside the model architecture.
          For example, the vector of the complete utterance and the intent is passed on to an embedding layer before
          they are compared and the loss is calculated.
        - ``number_of_transformer_layers``:
          This parameter sets the number of transformer layers to use (default: ``0``).
          The number of transformer layers corresponds to the transformer blocks to use for the model.
        - ``transformer_size``:
          This parameter sets the number of units in the transformer (default: ``None``).
          The vectors coming out of the transformers will have the given ``transformer_size``.
        - ``weight_sparsity``:
          This parameter defines the fraction of kernel weights that are set to 0 for all feed forward layers
          in the model (default: ``0.8``). The value should be between 0 and 1. If you set ``weight_sparsity``
          to 0, no kernel weights will be set to 0, the layer acts as a standard feed forward layer. You should not
          set ``weight_sparsity`` to 1 as this would result in all kernel weights being 0, i.e. the model is not able
          to learn.

    |

    In addition, the component can also be configured to train a response selector for a particular retrieval intent.
    The parameter ``retrieval_intent`` sets the name of the intent for which this response selector model is trained.
    Default is ``None``, i.e. the model is trained for all retrieval intents.

    |

    .. container:: toggle

        .. container:: header

            The above configuration parameters are the ones you should configure to fit your model to your data.
            However, additional parameters exist that can be adapted.

        .. code-block:: none

         +---------------------------------+-------------------+--------------------------------------------------------------+
         | Parameter                       | Default Value     | Description                                                  |
         +=================================+===================+==============================================================+
         | hidden_layers_sizes             | text: [256, 128]  | Hidden layer sizes for layers before the embedding layers    |
         |                                 | label: [256, 128] | for user messages and labels. The number of hidden layers is |
         |                                 |                   | equal to the length of the corresponding.                    |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | share_hidden_layers             | False             | Whether to share the hidden layer weights between user       |
         |                                 |                   | messages and labels.                                         |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | transformer_size                | None              | Number of units in transformer.                              |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | number_of_transformer_layers    | 0                 | Number of transformer layers.                                |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | number_of_attention_heads       | 4                 | Number of attention heads in transformer.                    |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | use_key_relative_attention      | False             | If 'True' use key relative embeddings in attention.          |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | use_value_relative_attention    | False             | If 'True' use value relative embeddings in attention.        |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | max_relative_position           | None              | Maximum position for relative embeddings.                    |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | unidirectional_encoder          | False             | Use a unidirectional or bidirectional encoder.               |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | batch_size                      | [64, 256]         | Initial and final value for batch sizes.                     |
         |                                 |                   | Batch size will be linearly increased for each epoch.        |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | batch_strategy                  | "balanced"        | Strategy used when creating batches.                         |
         |                                 |                   | Can be either 'sequence' or 'balanced'.                      |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | epochs                          | 300               | Number of epochs to train.                                   |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | random_seed                     | None              | Set random seed to any 'int' to get reproducible results.    |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | learning_rate                   | 0.001             | Initial learning rate for the optimizer.                     |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | embedding_dimension             | 20                | Dimension size of embedding vectors.                         |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | dense_dimension                 | text: 512         | Dense dimension for sparse features to use if no dense       |
         |                                 | label: 512        | features are present.                                        |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | concat_dimension                | text: 512         | Concat dimension for sequence and sentence features.         |
         |                                 | label: 512        |                                                              |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | number_of_negative_examples     | 20                | The number of incorrect labels. The algorithm will minimize  |
         |                                 |                   | their similarity to the user input during training.          |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | similarity_type                 | "auto"            | Type of similarity measure to use, either 'auto' or 'cosine' |
         |                                 |                   | or 'inner'.                                                  |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | loss_type                       | "softmax"         | The type of the loss function, either 'softmax' or 'margin'. |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | ranking_length                  | 10                | Number of top actions to normalize scores for loss type      |
         |                                 |                   | 'softmax'. Set to 0 to turn off normalization.               |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | maximum_positive_similarity     | 0.8               | Indicates how similar the algorithm should try to make       |
         |                                 |                   | embedding vectors for correct labels.                        |
         |                                 |                   | Should be 0.0 < ... < 1.0 for 'cosine' similarity type.      |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | maximum_negative_similarity     | -0.4              | Maximum negative similarity for incorrect labels.            |
         |                                 |                   | Should be -1.0 < ... < 1.0 for 'cosine' similarity type.     |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | use_maximum_negative_similarity | True              | If 'True' the algorithm only minimizes maximum similarity    |
         |                                 |                   | over incorrect intent labels, used only if 'loss_type' is    |
         |                                 |                   | set to 'margin'.                                             |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | scale_loss                      | True              | Scale loss inverse proportionally to confidence of correct   |
         |                                 |                   | prediction.                                                  |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | regularization_constant         | 0.002             | The scale of regularization.                                 |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | negative_margin_scale           | 0.8               | The scale of how important is to minimize the maximum        |
         |                                 |                   | similarity between embeddings of different labels.           |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | weight_sparsity                 | 0.8               | Sparsity of the weights in dense layers.                     |
         |                                 |                   | Value should be between 0 and 1.                             |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | drop_rate                       | 0.2               | Dropout rate for encoder. Value should be between 0 and 1.   |
         |                                 |                   | The higher the value the higher the regularization effect.   |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | drop_rate_attention             | 0.0               | Dropout rate for attention. Value should be between 0 and 1. |
         |                                 |                   | The higher the value the higher the regularization effect.   |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | use_sparse_input_dropout        | False             | If 'True' apply dropout to sparse input tensors.             |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | use_dense_input_dropout         | False             | If 'True' apply dropout to dense input tensors.              |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | evaluate_every_number_of_epochs | 20                | How often to calculate validation accuracy.                  |
         |                                 |                   | Set to '-1' to evaluate just once at the end of training.    |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | evaluate_on_number_of_examples  | 0                 | How many examples to use for hold out validation set.        |
         |                                 |                   | Large values may hurt performance, e.g. model accuracy.      |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | use_masked_language_model       | False             | If 'True' random tokens of the input message will be masked  |
         |                                 |                   | and the model should predict those tokens.                   |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | retrieval_intent                | None              | Name of the intent for which this response selector model is |
         |                                 |                   | trained.                                                     |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | tensorboard_log_directory       | None              | If you want to use tensorboard to visualize training         |
         |                                 |                   | metrics, set this option to a valid output directory. You    |
         |                                 |                   | can view the training metrics after training in tensorboard  |
         |                                 |                   | via 'tensorboard --logdir <path-to-given-directory>'.        |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | tensorboard_log_level           | "epoch"           | Define when training metrics for tensorboard should be       |
         |                                 |                   | logged. Either after every epoch ("epoch") or for every      |
         |                                 |                   | training step ("minibatch").                                 |
         +---------------------------------+-------------------+--------------------------------------------------------------+
         | featurizers                     | []                | List of featurizer names (alias names). Only features        |
         |                                 |                   | coming from the listed names are used. If list is empty      |
         |                                 |                   | all available features are used.                             |
         +---------------------------------+-------------------+--------------------------------------------------------------+

        .. note:: For ``cosine`` similarity ``maximum_positive_similarity`` and ``maximum_negative_similarity`` should
                  be between ``-1`` and ``1``.

        .. note:: There is an option to use linearly increasing batch size. The idea comes from
                  `<https://arxiv.org/abs/1711.00489>`_.
                  In order to do it pass a list to ``batch_size``, e.g. ``"batch_size": [64, 256]`` (default behavior).
                  If constant ``batch_size`` is required, pass an ``int``, e.g. ``"batch_size": 64``.

        .. note:: Parameter ``maximum_negative_similarity`` is set to a negative value to mimic the original
                  starspace algorithm in the case ``maximum_negative_similarity = maximum_positive_similarity``
                  and ``use_maximum_negative_similarity = False``.
                  See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.


Combined Entity Extractors and Intent Classifiers
-------------------------------------------------

.. _diet-classifier:

DIETClassifier
~~~~~~~~~~~~~~

:Short: Dual Intent Entity Transformer (DIET) used for intent classification and entity extraction
:Outputs: ``entities``, ``intent`` and ``intent_ranking``
:Requires: ``dense_features`` and/or ``sparse_features`` for user message and optionally the intent
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
    For the intent labels the transformer output for the complete utterance and intent labels are embedded into a
    single semantic vector space. We use the dot-product loss to maximize the similarity with the target label and
    minimize similarities with negative samples.

    If you want to learn more about the model, please take a look at our
    `videos <https://www.youtube.com/playlist?list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb>`__ where we explain the model
    architecture in detail.

    .. note:: If during prediction time a message contains **only** words unseen during training
              and no Out-Of-Vocabulary preprocessor was used, an empty intent ``None`` is predicted with confidence
              ``0.0``. This might happen if you only use the :ref:`CountVectorsFeaturizer` with a ``word`` analyzer
              as featurizer. If you use the ``char_wb`` analyzer, you should always get an intent with a confidence
              value ``> 0.0``.

:Configuration:

    If you want to use the ``DIETClassifier`` just for intent classification, set ``entity_recognition`` to ``False``.
    If you want to do only entity recognition, set ``intent_classification`` to ``False``.
    By default ``DIETClassifier`` does both, i.e. ``entity_recognition`` and ``intent_classification`` are set to
    ``True``.

    You can define a number of hyperparameters to adapt the model.
    If you want to adapt your model, start by modifying the following parameters:

        - ``epochs``:
          This parameter sets the number of times the algorithm will see the training data (default: ``300``).
          One ``epoch`` is equals to one forward pass and one backward pass of all the training examples.
          Sometimes the model needs more epochs to properly learn.
          Sometimes more epochs don't influence the performance.
          The lower the number of epochs the faster the model is trained.
        - ``hidden_layers_sizes``:
          This parameter allows you to define the number of feed forward layers and their output
          dimensions for user messages and intents (default: ``text: [], label: []``).
          Every entry in the list corresponds to a feed forward layer.
          For example, if you set ``text: [256, 128]``, we will add two feed forward layers in front of
          the transformer. The vectors of the input tokens (coming from the user message) will be passed on to those
          layers. The first layer will have an output dimension of 256 and the second layer will have an output
          dimension of 128. If an empty list is used (default behavior), no feed forward layer will be
          added.
          Make sure to use only positive integer values. Usually, numbers of power of two are used.
          Also, it is usual practice to have decreasing values in the list: next value is smaller or equal to the
          value before.
        - ``embedding_dimension``:
          This parameter defines the output dimension of the embedding layers used inside the model (default: ``20``).
          We are using multiple embeddings layers inside the model architecture.
          For example, the vector of the complete utterance and the intent is passed on to an embedding layer before
          they are compared and the loss is calculated.
        - ``number_of_transformer_layers``:
          This parameter sets the number of transformer layers to use (default: ``2``).
          The number of transformer layers corresponds to the transformer blocks to use for the model.
        - ``transformer_size``:
          This parameter sets the number of units in the transformer (default: ``256``).
          The vectors coming out of the transformers will have the given ``transformer_size``.
        - ``weight_sparsity``:
          This parameter defines the fraction of kernel weights that are set to 0 for all feed forward layers
          in the model (default: ``0.8``). The value should be between 0 and 1. If you set ``weight_sparsity``
          to 0, no kernel weights will be set to 0, the layer acts as a standard feed forward layer. You should not
          set ``weight_sparsity`` to 1 as this would result in all kernel weights being 0, i.e. the model is not able
          to learn.

    .. container:: toggle

        .. container:: header

            The above configuration parameters are the ones you should configure to fit your model to your data.
            However, additional parameters exist that can be adapted.

        .. code-block:: none

         +---------------------------------+------------------+--------------------------------------------------------------+
         | Parameter                       | Default Value    | Description                                                  |
         +=================================+==================+==============================================================+
         | hidden_layers_sizes             | text: []         | Hidden layer sizes for layers before the embedding layers    |
         |                                 | label: []        | for user messages and labels. The number of hidden layers is |
         |                                 |                  | equal to the length of the corresponding.                    |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | share_hidden_layers             | False            | Whether to share the hidden layer weights between user       |
         |                                 |                  | messages and labels.                                         |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | transformer_size                | 256              | Number of units in transformer.                              |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | number_of_transformer_layers    | 2                | Number of transformer layers.                                |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | number_of_attention_heads       | 4                | Number of attention heads in transformer.                    |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | use_key_relative_attention      | False            | If 'True' use key relative embeddings in attention.          |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | use_value_relative_attention    | False            | If 'True' use value relative embeddings in attention.        |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | max_relative_position           | None             | Maximum position for relative embeddings.                    |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | unidirectional_encoder          | False            | Use a unidirectional or bidirectional encoder.               |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | batch_size                      | [64, 256]        | Initial and final value for batch sizes.                     |
         |                                 |                  | Batch size will be linearly increased for each epoch.        |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | batch_strategy                  | "balanced"       | Strategy used when creating batches.                         |
         |                                 |                  | Can be either 'sequence' or 'balanced'.                      |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | epochs                          | 300              | Number of epochs to train.                                   |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | random_seed                     | None             | Set random seed to any 'int' to get reproducible results.    |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | learning_rate                   | 0.001            | Initial learning rate for the optimizer.                     |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | embedding_dimension             | 20               | Dimension size of embedding vectors.                         |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | dense_dimension                 | text: 512        | Dense dimension for sparse features to use if no dense       |
         |                                 | label: 20        | features are present.                                        |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | concat_dimension                | text: 512        | Concat dimension for sequence and sentence features.         |
         |                                 | label: 20        |                                                              |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | number_of_negative_examples     | 20               | The number of incorrect labels. The algorithm will minimize  |
         |                                 |                  | their similarity to the user input during training.          |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | similarity_type                 | "auto"           | Type of similarity measure to use, either 'auto' or 'cosine' |
         |                                 |                  | or 'inner'.                                                  |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | loss_type                       | "softmax"        | The type of the loss function, either 'softmax' or 'margin'. |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | ranking_length                  | 10               | Number of top actions to normalize scores for loss type      |
         |                                 |                  | 'softmax'. Set to 0 to turn off normalization.               |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | maximum_positive_similarity     | 0.8              | Indicates how similar the algorithm should try to make       |
         |                                 |                  | embedding vectors for correct labels.                        |
         |                                 |                  | Should be 0.0 < ... < 1.0 for 'cosine' similarity type.      |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | maximum_negative_similarity     | -0.4             | Maximum negative similarity for incorrect labels.            |
         |                                 |                  | Should be -1.0 < ... < 1.0 for 'cosine' similarity type.     |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | use_maximum_negative_similarity | True             | If 'True' the algorithm only minimizes maximum similarity    |
         |                                 |                  | over incorrect intent labels, used only if 'loss_type' is    |
         |                                 |                  | set to 'margin'.                                             |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | scale_loss                      | False            | Scale loss inverse proportionally to confidence of correct   |
         |                                 |                  | prediction.                                                  |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | regularization_constant         | 0.002            | The scale of regularization.                                 |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | negative_margin_scale           | 0.8              | The scale of how important it is to minimize the maximum     |
         |                                 |                  | similarity between embeddings of different labels.           |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | weight_sparsity                 | 0.8              | Sparsity of the weights in dense layers.                     |
         |                                 |                  | Value should be between 0 and 1.                             |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | drop_rate                       | 0.2              | Dropout rate for encoder. Value should be between 0 and 1.   |
         |                                 |                  | The higher the value the higher the regularization effect.   |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | drop_rate_attention             | 0.0              | Dropout rate for attention. Value should be between 0 and 1. |
         |                                 |                  | The higher the value the higher the regularization effect.   |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | use_sparse_input_dropout        | True             | If 'True' apply dropout to sparse input tensors.             |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | use_dense_input_dropout         | True             | If 'True' apply dropout to dense input tensors.              |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | evaluate_every_number_of_epochs | 20               | How often to calculate validation accuracy.                  |
         |                                 |                  | Set to '-1' to evaluate just once at the end of training.    |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | evaluate_on_number_of_examples  | 0                | How many examples to use for hold out validation set.        |
         |                                 |                  | Large values may hurt performance, e.g. model accuracy.      |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | intent_classification           | True             | If 'True' intent classification is trained and intents are   |
         |                                 |                  | predicted.                                                   |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | entity_recognition              | True             | If 'True' entity recognition is trained and entities are     |
         |                                 |                  | extracted.                                                   |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | use_masked_language_model       | False            | If 'True' random tokens of the input message will be masked  |
         |                                 |                  | and the model has to predict those tokens. It acts like a    |
         |                                 |                  | regularizer and should help to learn a better contextual     |
         |                                 |                  | representation of the input.                                 |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | tensorboard_log_directory       | None             | If you want to use tensorboard to visualize training         |
         |                                 |                  | metrics, set this option to a valid output directory. You    |
         |                                 |                  | can view the training metrics after training in tensorboard  |
         |                                 |                  | via 'tensorboard --logdir <path-to-given-directory>'.        |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | tensorboard_log_level           | "epoch"          | Define when training metrics for tensorboard should be       |
         |                                 |                  | logged. Either after every epoch ('epoch') or for every      |
         |                                 |                  | training step ('minibatch').                                 |
         +---------------------------------+------------------+--------------------------------------------------------------+
         | featurizers                     | []               | List of featurizer names (alias names). Only features        |
         |                                 |                  | coming from the listed names are used. If list is empty      |
         |                                 |                  | all available features are used.                             |
         +---------------------------------+------------------+--------------------------------------------------------------+

        .. note:: For ``cosine`` similarity ``maximum_positive_similarity`` and ``maximum_negative_similarity`` should
                  be between ``-1`` and ``1``.

        .. note:: There is an option to use linearly increasing batch size. The idea comes from
                  `<https://arxiv.org/abs/1711.00489>`_.
                  In order to do it pass a list to ``batch_size``, e.g. ``"batch_size": [64, 256]`` (default behavior).
                  If constant ``batch_size`` is required, pass an ``int``, e.g. ``"batch_size": 64``.

        .. note:: Parameter ``maximum_negative_similarity`` is set to a negative value to mimic the original
                  starspace algorithm in the case ``maximum_negative_similarity = maximum_positive_similarity``
                  and ``use_maximum_negative_similarity = False``.
                  See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details.
