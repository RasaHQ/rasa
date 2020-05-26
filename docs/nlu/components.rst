:desc: Customize the components and parameters of Rasa's Machine Learning based a
       Natural Language Understanding pipeline a
 a
.. _components: a
 a
Components a
========== a
 a
.. edit-link:: a
 a
This is a reference of the configuration options for every built-in a
component in Rasa Open Source. If you want to build a custom component, check a
out :ref:`custom-nlu-components`. a
 a
.. contents:: a
   :local: a
 a
 a
Word Vector Sources a
------------------- a
 a
The following components load pre-trained models that are needed if you want to use pre-trained a
word vectors in your pipeline. a
 a
.. _MitieNLP: a
 a
MitieNLP a
~~~~~~~~ a
 a
:Short: MITIE initializer a
:Outputs: Nothing a
:Requires: Nothing a
:Description: a
    Initializes MITIE structures. Every MITIE component relies on this, a
    hence this should be put at the beginning a
    of every pipeline that uses any MITIE components. a
:Configuration: a
    The MITIE library needs a language model file, that **must** be specified in a
    the configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "MitieNLP" a
          # language model to load a
          model: "data/total_word_feature_extractor.dat" a
 a
    For more information where to get that file from, head over to a
    :ref:`installing MITIE <install-mitie>`. a
 a
.. _SpacyNLP: a
 a
SpacyNLP a
~~~~~~~~ a
 a
:Short: spaCy language initializer a
:Outputs: Nothing a
:Requires: Nothing a
:Description: a
    Initializes spaCy structures. Every spaCy component relies on this, hence this should be put at the beginning a
    of every pipeline that uses any spaCy components. a
:Configuration: a
    You need to specify the language model to use. a
    By default the language configured in the pipeline will be used as the language model name. a
    If the spaCy model to be used has a name that is different from the language tag (``"en"``, ``"de"``, etc.), a
    the model name can be specified using the configuration variable ``model``. a
    The name will be passed to ``spacy.load(name)``. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "SpacyNLP" a
          # language model to load a
          model: "en_core_web_md" a
 a
          # when retrieving word vectors, this will decide if the casing a
          # of the word is relevant. E.g. `hello` and `Hello` will a
          # retrieve the same vector, if set to `False`. For some a
          # applications and models it makes sense to differentiate a
          # between these two words, therefore setting this to `True`. a
          case_sensitive: False a
 a
    For more information on how to download the spaCy models, head over to a
    :ref:`installing SpaCy <install-spacy>`. a
 a
.. _HFTransformersNLP: a
 a
HFTransformersNLP a
~~~~~~~~~~~~~~~~~ a
 a
:Short: HuggingFace's Transformers based pre-trained language model initializer a
:Outputs: Nothing a
:Requires: Nothing a
:Description: a
    Initializes specified pre-trained language model from HuggingFace's `Transformers library a
    <https://huggingface.co/transformers/>`__.  The component applies language model specific tokenization and a
    featurization to compute sequence and sentence level representations for each example in the training data. a
    Include :ref:`LanguageModelTokenizer` and :ref:`LanguageModelFeaturizer` to utilize the output of this a
    component for downstream NLU models. a
 a
     .. note:: To use ``HFTransformersNLP`` component, install Rasa Open Source with ``pip install rasa[transformers]``. a
 a
:Configuration: a
    You should specify what language model to load via the parameter ``model_name``. See the below table for the a
    available language models. a
    Additionally, you can also specify the architecture variation of the chosen language model by specifying the a
    parameter ``model_weights``. a
    The full list of supported architectures can be found a
    `here <https://huggingface.co/transformers/pretrained_models.html>`__. a
    If left empty, it uses the default model architecture that original Transformers library loads (see table below). a
 a
    .. code-block:: none a
 a
        +----------------+--------------+-------------------------+ a
        | Language Model | Parameter    | Default value for       | a
        |                | "model_name" | "model_weights"         | a
        +----------------+--------------+-------------------------+ a
        | BERT           | bert         | bert-base-uncased       | a
        +----------------+--------------+-------------------------+ a
        | GPT            | gpt          | openai-gpt              | a
        +----------------+--------------+-------------------------+ a
        | GPT-2          | gpt2         | gpt2                    | a
        +----------------+--------------+-------------------------+ a
        | XLNet          | xlnet        | xlnet-base-cased        | a
        +----------------+--------------+-------------------------+ a
        | DistilBERT     | distilbert   | distilbert-base-uncased | a
        +----------------+--------------+-------------------------+ a
        | RoBERTa        | roberta      | roberta-base            | a
        +----------------+--------------+-------------------------+ a
 a
    The following configuration loads the language model BERT: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
          - name: HFTransformersNLP a
            # Name of the language model to use a
            model_name: "bert" a
            # Pre-Trained weights to be loaded a
            model_weights: "bert-base-uncased" a
             a
            # An optional path to a specific directory to download and cache the pre-trained model weights. a
            # The `default` cache_dir is the same as https://huggingface.co/transformers/serialization.html#cache-directory . a
            cache_dir: null a
 a
.. _tokenizers: a
 a
Tokenizers a
---------- a
 a
Tokenizers split text into tokens. a
If you want to split intents into multiple labels, e.g. for predicting multiple intents or for a
modeling hierarchical intent structure, use the following flags with any tokenizer: a
 a
- ``intent_tokenization_flag`` indicates whether to tokenize intent labels or not. Set it to ``True``, so that intent a
  labels are tokenized. a
- ``intent_split_symbol`` sets the delimiter string to split the intent labels, default is underscore a
  (``_``). a
 a
    .. note:: All tokenizers add an additional token ``__CLS__`` to the end of the list of tokens when tokenizing a
              text and responses. a
 a
.. _WhitespaceTokenizer: a
 a
WhitespaceTokenizer a
~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Tokenizer using whitespaces as a separator a
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified) a
:Requires: Nothing a
:Description: a
    Creates a token for every whitespace separated character sequence. a
:Configuration: a
    Make the tokenizer case insensitive by adding the ``case_sensitive: False`` option, the a
    default being ``case_sensitive: True``. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "WhitespaceTokenizer" a
          # Flag to check whether to split intents a
          "intent_tokenization_flag": False a
          # Symbol on which intent should be split a
          "intent_split_symbol": "_" a
          # Text will be tokenized with case sensitive as default a
          "case_sensitive": True a
 a
 a
JiebaTokenizer a
~~~~~~~~~~~~~~ a
 a
:Short: Tokenizer using Jieba for Chinese language a
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified) a
:Requires: Nothing a
:Description: a
    Creates tokens using the Jieba tokenizer specifically for Chinese a
    language. It will only work for the Chinese language. a
 a
    .. note:: a
        To use ``JiebaTokenizer`` you need to install Jieba with ``pip install jieba``. a
 a
:Configuration: a
    User's custom dictionary files can be auto loaded by specifying the files' directory path via ``dictionary_path``. a
    If the ``dictionary_path`` is ``None`` (the default), then no custom dictionary will be used. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "JiebaTokenizer" a
          dictionary_path: "path/to/custom/dictionary/dir" a
          # Flag to check whether to split intents a
          "intent_tokenization_flag": False a
          # Symbol on which intent should be split a
          "intent_split_symbol": "_" a
 a
 a
MitieTokenizer a
~~~~~~~~~~~~~~ a
 a
:Short: Tokenizer using MITIE a
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified) a
:Requires: :ref:`MitieNLP` a
:Description: Creates tokens using the MITIE tokenizer. a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "MitieTokenizer" a
          # Flag to check whether to split intents a
          "intent_tokenization_flag": False a
          # Symbol on which intent should be split a
          "intent_split_symbol": "_" a
 a
SpacyTokenizer a
~~~~~~~~~~~~~~ a
 a
:Short: Tokenizer using spaCy a
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified) a
:Requires: :ref:`SpacyNLP` a
:Description: a
    Creates tokens using the spaCy tokenizer. a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "SpacyTokenizer" a
          # Flag to check whether to split intents a
          "intent_tokenization_flag": False a
          # Symbol on which intent should be split a
          "intent_split_symbol": "_" a
 a
.. _ConveRTTokenizer: a
 a
ConveRTTokenizer a
~~~~~~~~~~~~~~~~ a
 a
:Short: Tokenizer using `ConveRT <https://github.com/PolyAI-LDN/polyai-models#convert>`__ model. a
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified) a
:Requires: Nothing a
:Description: a
    Creates tokens using the ConveRT tokenizer. Must be used whenever the :ref:`ConveRTFeaturizer` is used. a
 a
    .. note:: a
        Since ``ConveRT`` model is trained only on an English corpus of conversations, this tokenizer should only a
        be used if your training data is in English language. a
 a
    .. note:: a
        To use ``ConveRTTokenizer``, install Rasa Open Source with ``pip install rasa[convert]``. a
 a
 a
:Configuration: a
    Make the tokenizer case insensitive by adding the ``case_sensitive: False`` option, the a
    default being ``case_sensitive: True``. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "ConveRTTokenizer" a
          # Flag to check whether to split intents a
          "intent_tokenization_flag": False a
          # Symbol on which intent should be split a
          "intent_split_symbol": "_" a
          # Text will be tokenized with case sensitive as default a
          "case_sensitive": True a
 a
.. _LanguageModelTokenizer: a
 a
LanguageModelTokenizer a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Tokenizer from pre-trained language models a
:Outputs: ``tokens`` for user messages, responses (if present), and intents (if specified) a
:Requires: :ref:`HFTransformersNLP` a
:Description: a
    Creates tokens using the pre-trained language model specified in upstream :ref:`HFTransformersNLP` component. a
    Must be used whenever the :ref:`LanguageModelFeaturizer` is used. a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "LanguageModelTokenizer" a
          # Flag to check whether to split intents a
          "intent_tokenization_flag": False a
          # Symbol on which intent should be split a
          "intent_split_symbol": "_" a
 a
 a
 a
.. _text-featurizers: a
 a
Text Featurizers a
---------------- a
 a
Text featurizers are divided into two different categories: sparse featurizers and dense featurizers. a
Sparse featurizers are featurizers that return feature vectors with a lot of missing values, e.g. zeros. a
As those feature vectors would normally take up a lot of memory, we store them as sparse features. a
Sparse features only store the values that are non zero and their positions in the vector. a
Thus, we save a lot of memory and are able to train on larger datasets. a
 a
By default all featurizers will return a matrix of length ``(number-of-tokens x feature-dimension)``. a
So, the returned matrix will have a feature vector for every token. a
This allows us to train sequence models. a
However, the additional token at the end (e.g. ``__CLS__``) contains features for the complete utterance. a
This feature vector can be used in any bag-of-words model. a
The corresponding classifier can therefore decide what kind of features to use. a
 a
 a
.. _MitieFeaturizer: a
 a
MitieFeaturizer a
~~~~~~~~~~~~~~~ a
 a
:Short: a
    Creates a vector representation of user message and response (if specified) using the MITIE featurizer. a
:Outputs: ``dense_features`` for user messages and responses a
:Requires: :ref:`MitieNLP` a
:Type: Dense featurizer a
:Description: a
    Creates features for entity extraction, intent classification, and response classification using the MITIE a
    featurizer. a
 a
    .. note:: a
 a
        NOT used by the ``MitieIntentClassifier`` component. But can be used by any component later in the pipeline a
        that makes use of ``dense_features``. a
 a
:Configuration: a
    The sentence vector, i.e. the vector of the ``__CLS__`` token, can be calculated in two different ways, either via a
    mean or via max pooling. You can specify the pooling method in your configuration file with the option ``pooling``. a
    The default pooling method is set to ``mean``. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "MitieFeaturizer" a
          # Specify what pooling operation should be used to calculate the vector of a
          # the __CLS__ token. Available options: 'mean' and 'max'. a
          "pooling": "mean" a
 a
 a
.. _SpacyFeaturizer: a
 a
SpacyFeaturizer a
~~~~~~~~~~~~~~~ a
 a
:Short: a
    Creates a vector representation of user message and response (if specified) using the spaCy featurizer. a
:Outputs: ``dense_features`` for user messages and responses a
:Requires: :ref:`SpacyNLP` a
:Type: Dense featurizer a
:Description: a
    Creates features for entity extraction, intent classification, and response classification using the spaCy a
    featurizer. a
:Configuration: a
    The sentence vector, i.e. the vector of the ``__CLS__`` token, can be calculated in two different ways, either via a
    mean or via max pooling. You can specify the pooling method in your configuration file with the option ``pooling``. a
    The default pooling method is set to ``mean``. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "SpacyFeaturizer" a
          # Specify what pooling operation should be used to calculate the vector of a
          # the __CLS__ token. Available options: 'mean' and 'max'. a
          "pooling": "mean" a
 a
 a
.. _ConveRTFeaturizer: a
 a
ConveRTFeaturizer a
~~~~~~~~~~~~~~~~~ a
 a
:Short: a
    Creates a vector representation of user message and response (if specified) using a
    `ConveRT <https://github.com/PolyAI-LDN/polyai-models>`__ model. a
:Outputs: ``dense_features`` for user messages and responses a
:Requires: :ref:`ConveRTTokenizer` a
:Type: Dense featurizer a
:Description: a
    Creates features for entity extraction, intent classification, and response selection. a
    It uses the `default signature <https://github.com/PolyAI-LDN/polyai-models#tfhub-signatures>`_ to compute vector a
    representations of input text. a
 a
    .. note:: a
        Since ``ConveRT`` model is trained only on an English corpus of conversations, this featurizer should only a
        be used if your training data is in English language. a
 a
    .. note:: a
        To use ``ConveRTTokenizer``, install Rasa Open Source with ``pip install rasa[convert]``. a
 a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "ConveRTFeaturizer" a
 a
 a
.. _LanguageModelFeaturizer: a
 a
LanguageModelFeaturizer a
~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: a
    Creates a vector representation of user message and response (if specified) using a pre-trained language model. a
:Outputs: ``dense_features`` for user messages and responses a
:Requires: :ref:`HFTransformersNLP` and :ref:`LanguageModelTokenizer` a
:Type: Dense featurizer a
:Description: a
    Creates features for entity extraction, intent classification, and response selection. a
    Uses the pre-trained language model specified in upstream :ref:`HFTransformersNLP` component to compute vector a
    representations of input text. a
 a
    .. note:: a
        Please make sure that you use a language model which is pre-trained on the same language corpus as that of your a
        training data. a
 a
:Configuration: a
 a
    Include :ref:`HFTransformersNLP` and :ref:`LanguageModelTokenizer` components before this component. Use a
    :ref:`LanguageModelTokenizer` to ensure tokens are correctly set for all components throughout the pipeline. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "LanguageModelFeaturizer" a
 a
 a
.. _RegexFeaturizer: a
 a
RegexFeaturizer a
~~~~~~~~~~~~~~~ a
 a
:Short: Creates a vector representation of user message using regular expressions. a
:Outputs: ``sparse_features`` for user messages and ``tokens.pattern`` a
:Requires: ``tokens`` a
:Type: Sparse featurizer a
:Description: a
    Creates features for entity extraction and intent classification. a
    During training the ``RegexFeaturizer`` creates a list of regular expressions defined in the training a
    data format. a
    For each regex, a feature will be set marking whether this expression was found in the user message or not. a
    All features will later be fed into an intent classifier / entity extractor to simplify classification (assuming a
    the classifier has learned during the training phase, that this set feature indicates a certain intent / entity). a
    Regex features for entity extraction are currently only supported by the :ref:`CRFEntityExtractor` and the a
    :ref:`diet-classifier` components! a
 a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "RegexFeaturizer" a
 a
.. _CountVectorsFeaturizer: a
 a
CountVectorsFeaturizer a
~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Creates bag-of-words representation of user messages, intents, and responses. a
:Outputs: ``sparse_features`` for user messages, intents, and responses a
:Requires: ``tokens`` a
:Type: Sparse featurizer a
:Description: a
    Creates features for intent classification and response selection. a
    Creates bag-of-words representation of user message, intent, and response using a
    `sklearn's CountVectorizer <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_. a
    All tokens which consist only of digits (e.g. 123 and 99 but not a123d) will be assigned to the same feature. a
 a
:Configuration: a
    See `sklearn's CountVectorizer docs <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_ a
    for detailed description of the configuration parameters. a
 a
    This featurizer can be configured to use word or character n-grams, using the ``analyzer`` configuration parameter. a
    By default ``analyzer`` is set to ``word`` so word token counts are used as features. a
    If you want to use character n-grams, set ``analyzer`` to ``char`` or ``char_wb``. a
    The lower and upper boundaries of the n-grams can be configured via the parameters ``min_ngram`` and ``max_ngram``. a
    By default both of them are set to ``1``. a
 a
    .. note:: a
        Option ``char_wb`` creates character n-grams only from text inside word boundaries; a
        n-grams at the edges of words are padded with space. a
        This option can be used to create `Subword Semantic Hashing <https://arxiv.org/abs/1810.07150>`_. a
 a
    .. note:: a
        For character n-grams do not forget to increase ``min_ngram`` and ``max_ngram`` parameters. a
        Otherwise the vocabulary will contain only single letters. a
 a
    Handling Out-Of-Vocabulary (OOV) words: a
 a
        .. note:: Enabled only if ``analyzer`` is ``word``. a
 a
        Since the training is performed on limited vocabulary data, it cannot be guaranteed that during prediction a
        an algorithm will not encounter an unknown word (a word that were not seen during training). a
        In order to teach an algorithm how to treat unknown words, some words in training data can be substituted a
        by generic word ``OOV_token``. a
        In this case during prediction all unknown words will be treated as this generic word ``OOV_token``. a
 a
        For example, one might create separate intent ``outofscope`` in the training data containing messages of a
        different number of ``OOV_token`` s and maybe some additional general words. a
        Then an algorithm will likely classify a message with unknown words as this intent ``outofscope``. a
 a
        You can either set the ``OOV_token`` or a list of words ``OOV_words``: a
 a
            - ``OOV_token`` set a keyword for unseen words; if training data contains ``OOV_token`` as words in some a
              messages, during prediction the words that were not seen during training will be substituted with a
              provided ``OOV_token``; if ``OOV_token=None`` (default behaviour) words that were not seen during a
              training will be ignored during prediction time; a
            - ``OOV_words`` set a list of words to be treated as ``OOV_token`` during training; if a list of words a
              that should be treated as Out-Of-Vocabulary is known, it can be set to ``OOV_words`` instead of manually a
              changing it in training data or using custom preprocessor. a
 a
        .. note:: a
            This featurizer creates a bag-of-words representation by **counting** words, a
            so the number of ``OOV_token`` in the sentence might be important. a
 a
        .. note:: a
            Providing ``OOV_words`` is optional, training data can contain ``OOV_token`` input manually or by custom a
            additional preprocessor. a
            Unseen words will be substituted with ``OOV_token`` **only** if this token is present in the training a
            data or ``OOV_words`` list is provided. a
 a
    If you want to share the vocabulary between user messages and intents, you need to set the option a
    ``use_shared_vocab`` to ``True``. In that case a common vocabulary set between tokens in intents and user messages a
    is build. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "CountVectorsFeaturizer" a
          # Analyzer to use, either 'word', 'char', or 'char_wb' a
          "analyzer": "word" a
          # Set the lower and upper boundaries for the n-grams a
          "min_ngram": 1 a
          "max_ngram": 1 a
          # Set the out-of-vocabulary token a
          "OOV_token": "_oov_" a
          # Whether to use a shared vocab a
          "use_shared_vocab": False a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            The above configuration parameters are the ones you should configure to fit your model to your data. a
            However, additional parameters exist that can be adapted. a
 a
        .. code-block:: none a
 a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | Parameter         | Default Value     | Description                                                  | a
         +===================+===================+==============================================================+ a
         | use_shared_vocab  | False             | If set to 'True' a common vocabulary is used for labels      | a
         |                   |                   | and user message.                                            | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | analyzer          | word              | Whether the features should be made of word n-gram or        | a
         |                   |                   | character n-grams. Option ‘char_wb’ creates character        | a
         |                   |                   | n-grams only from text inside word boundaries;               | a
         |                   |                   | n-grams at the edges of words are padded with space.         | a
         |                   |                   | Valid values: 'word', 'char', 'char_wb'.                     | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | token_pattern     | r"(?u)\b\w\w+\b"  | Regular expression used to detect tokens.                    | a
         |                   |                   | Only used if 'analyzer' is set to 'word'.                    | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | strip_accents     | None              | Remove accents during the pre-processing step.               | a
         |                   |                   | Valid values: 'ascii', 'unicode', 'None'.                    | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | stop_words        | None              | A list of stop words to use.                                 | a
         |                   |                   | Valid values: 'english' (uses an internal list of            | a
         |                   |                   | English stop words), a list of custom stop words, or         | a
         |                   |                   | 'None'.                                                      | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | min_df            | 1                 | When building the vocabulary ignore terms that have a        | a
         |                   |                   | document frequency strictly lower than the given threshold.  | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | max_df            | 1                 | When building the vocabulary ignore terms that have a        | a
         |                   |                   | document frequency strictly higher than the given threshold  | a
         |                   |                   | (corpus-specific stop words).                                | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | min_ngram         | 1                 | The lower boundary of the range of n-values for different    | a
         |                   |                   | word n-grams or char n-grams to be extracted.                | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | max_ngram         | 1                 | The upper boundary of the range of n-values for different    | a
         |                   |                   | word n-grams or char n-grams to be extracted.                | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | max_features      | None              | If not 'None', build a vocabulary that only consider the top | a
         |                   |                   | max_features ordered by term frequency across the corpus.    | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | lowercase         | True              | Convert all characters to lowercase before tokenizing.       | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | OOV_token         | None              | Keyword for unseen words.                                    | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
         | OOV_words         | []                | List of words to be treated as 'OOV_token' during training.  | a
         +-------------------+-------------------+--------------------------------------------------------------+ a
 a
 a
.. _LexicalSyntacticFeaturizer: a
 a
LexicalSyntacticFeaturizer a
~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Creates lexical and syntactic features for a user message to support entity extraction. a
:Outputs: ``sparse_features`` for user messages a
:Requires: ``tokens`` a
:Type: Sparse featurizer a
:Description: a
    Creates features for entity extraction. a
    Moves with a sliding window over every token in the user message and creates features according to the a
    configuration (see below). As a default configuration is present, you don't need to specify a configuration. a
:Configuration: a
    You can configure what kind of lexical and syntactic features the featurizer should extract. a
    The following features are available: a
 a
    .. code-block:: none a
 a
        ==============  ========================================================================================== a
        Feature Name    Description a
        ==============  ========================================================================================== a
        BOS             Checks if the token is at the beginning of the sentence. a
        EOS             Checks if the token is at the end of the sentence. a
        low             Checks if the token is lower case. a
        upper           Checks if the token is upper case. a
        title           Checks if the token starts with an uppercase character and all remaining characters are a
                        lowercased. a
        digit           Checks if the token contains just digits. a
        prefix5         Take the first five characters of the token. a
        prefix2         Take the first two characters of the token. a
        suffix5         Take the last five characters of the token. a
        suffix3         Take the last three characters of the token. a
        suffix2         Take the last two characters of the token. a
        suffix1         Take the last character of the token. a
        pos             Take the Part-of-Speech tag of the token (``SpacyTokenizer`` required). a
        pos2            Take the first two characters of the Part-of-Speech tag of the token a
                        (``SpacyTokenizer`` required). a
        ==============  ========================================================================================== a
 a
    As the featurizer is moving over the tokens in a user message with a sliding window, you can define features for a
    previous tokens, the current token, and the next tokens in the sliding window. a
    You define the features as a [before, token, after] array. a
    If you want to define features for the token before, the current token, and the token after, a
    your features configuration would look like this: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: LexicalSyntacticFeaturizer a
          "features": [ a
            ["low", "title", "upper"], a
            ["BOS", "EOS", "low", "upper", "title", "digit"], a
            ["low", "title", "upper"], a
          ] a
 a
    This configuration is also the default configuration. a
 a
    .. note:: If you want to make use of ``pos`` or ``pos2`` you need to add ``SpacyTokenizer`` to your pipeline. a
 a
 a
Intent Classifiers a
------------------ a
 a
Intent classifiers assign one of the intents defined in the domain file to incoming user messages. a
 a
MitieIntentClassifier a
~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: a
    MITIE intent classifier (using a a
    `text categorizer <https://github.com/mit-nlp/MITIE/blob/master/examples/python/text_categorizer_pure_model.py>`_) a
:Outputs: ``intent`` a
:Requires: ``tokens`` for user message and :ref:`MitieNLP` a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "intent": {"name": "greet", "confidence": 0.98343} a
        } a
 a
:Description: a
    This classifier uses MITIE to perform intent classification. The underlying classifier a
    is using a multi-class linear SVM with a sparse linear kernel (see a
    `MITIE trainer code <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/text_categorizer_trainer.cpp#L222>`_). a
 a
    .. note:: This classifier does not rely on any featurizer as it extracts features on its own. a
 a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "MitieIntentClassifier" a
 a
SklearnIntentClassifier a
~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Sklearn intent classifier a
:Outputs: ``intent`` and ``intent_ranking`` a
:Requires: ``dense_features`` for user messages a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "intent": {"name": "greet", "confidence": 0.78343}, a
            "intent_ranking": [ a
                { a
                    "confidence": 0.1485910906220309, a
                    "name": "goodbye" a
                }, a
                { a
                    "confidence": 0.08161531595656784, a
                    "name": "restaurant_search" a
                } a
            ] a
        } a
 a
:Description: a
    The sklearn intent classifier trains a linear SVM which gets optimized using a grid search. It also provides a
    rankings of the labels that did not "win". The ``SklearnIntentClassifier`` needs to be preceded by a dense a
    featurizer in the pipeline. This dense featurizer creates the features used for the classification. a
    For more information about the algorithm itself, take a look at the a
    `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`__ a
    documentation. a
 a
:Configuration: a
    During the training of the SVM a hyperparameter search is run to find the best parameter set. a
    In the configuration you can specify the parameters that will get tried. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "SklearnIntentClassifier" a
          # Specifies the list of regularization values to a
          # cross-validate over for C-SVM. a
          # This is used with the ``kernel`` hyperparameter in GridSearchCV. a
          C: [1, 2, 5, 10, 20, 100] a
          # Specifies the kernel to use with C-SVM. a
          # This is used with the ``C`` hyperparameter in GridSearchCV. a
          kernels: ["linear"] a
          # Gamma parameter of the C-SVM. a
          "gamma": [0.1] a
          # We try to find a good number of cross folds to use during a
          # intent training, this specifies the max number of folds. a
          "max_cross_validation_folds": 5 a
          # Scoring function used for evaluating the hyper parameters. a
          # This can be a name or a function. a
          "scoring_function": "f1_weighted" a
 a
.. _keyword_intent_classifier: a
 a
KeywordIntentClassifier a
~~~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Simple keyword matching intent classifier, intended for small, short-term projects. a
:Outputs: ``intent`` a
:Requires: Nothing a
 a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "intent": {"name": "greet", "confidence": 1.0} a
        } a
 a
:Description: a
    This classifier works by searching a message for keywords. a
    The matching is case sensitive by default and searches only for exact matches of the keyword-string in the user a
    message. a
    The keywords for an intent are the examples of that intent in the NLU training data. a
    This means the entire example is the keyword, not the individual words in the example. a
 a
    .. note:: This classifier is intended only for small projects or to get started. If a
              you have few NLU training data, you can take a look at the recommended pipelines in a
              :ref:`choosing-a-pipeline`. a
 a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "KeywordIntentClassifier" a
          case_sensitive: True a
 a
 a
DIETClassifier a
~~~~~~~~~~~~~~ a
 a
:Short: Dual Intent Entity Transformer (DIET) used for intent classification and entity extraction a
:Description: a
    You can find the detailed description of the :ref:`diet-classifier` under the section a
    `Combined Entity Extractors and Intent Classifiers`. a
 a
Entity Extractors a
----------------- a
 a
Entity extractors extract entities, such as person names or locations, from the user message. a
 a
MitieEntityExtractor a
~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: MITIE entity extraction (using a `MITIE NER trainer <https://github.com/mit-nlp/MITIE/blob/master/mitielib/src/ner_trainer.cpp>`_) a
:Outputs: ``entities`` a
:Requires: :ref:`MitieNLP` and ``tokens`` a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "entities": [{ a
                "value": "New York City", a
                "start": 20, a
                "end": 33, a
                "confidence": null, a
                "entity": "city", a
                "extractor": "MitieEntityExtractor" a
            }] a
        } a
 a
:Description: a
    ``MitieEntityExtractor`` uses the MITIE entity extraction to find entities in a message. The underlying classifier a
    is using a multi class linear SVM with a sparse linear kernel and custom features. a
    The MITIE component does not provide entity confidence values. a
 a
    .. note:: This entity extractor does not rely on any featurizer as it extracts features on its own. a
 a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "MitieEntityExtractor" a
 a
.. _SpacyEntityExtractor: a
 a
SpacyEntityExtractor a
~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: spaCy entity extraction a
:Outputs: ``entities`` a
:Requires: :ref:`SpacyNLP` a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "entities": [{ a
                "value": "New York City", a
                "start": 20, a
                "end": 33, a
                "confidence": null, a
                "entity": "city", a
                "extractor": "SpacyEntityExtractor" a
            }] a
        } a
 a
:Description: a
    Using spaCy this component predicts the entities of a message. spaCy uses a statistical BILOU transition model. a
    As of now, this component can only use the spaCy builtin entity extraction models and can not be retrained. a
    This extractor does not provide any confidence scores. a
 a
:Configuration: a
    Configure which dimensions, i.e. entity types, the spaCy component a
    should extract. A full list of available dimensions can be found in a
    the `spaCy documentation <https://spacy.io/api/annotation#section-named-entities>`_. a
    Leaving the dimensions option unspecified will extract all available dimensions. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "SpacyEntityExtractor" a
          # dimensions to extract a
          dimensions: ["PERSON", "LOC", "ORG", "PRODUCT"] a
 a
 a
EntitySynonymMapper a
~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Maps synonymous entity values to the same value. a
:Outputs: Modifies existing entities that previous entity extraction components found. a
:Requires: Nothing a
:Description: a
    If the training data contains defined synonyms, this component will make sure that detected entity values will a
    be mapped to the same value. For example, if your training data contains the following examples: a
 a
    .. code-block:: json a
 a
        [ a
            { a
              "text": "I moved to New York City", a
              "intent": "inform_relocation", a
              "entities": [{ a
                "value": "nyc", a
                "start": 11, a
                "end": 24, a
                "entity": "city", a
              }] a
            }, a
            { a
              "text": "I got a new flat in NYC.", a
              "intent": "inform_relocation", a
              "entities": [{ a
                "value": "nyc", a
                "start": 20, a
                "end": 23, a
                "entity": "city", a
              }] a
            } a
        ] a
 a
    This component will allow you to map the entities ``New York City`` and ``NYC`` to ``nyc``. The entity a
    extraction will return ``nyc`` even though the message contains ``NYC``. When this component changes an a
    existing entity, it appends itself to the processor list of this entity. a
 a
:Configuration: a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "EntitySynonymMapper" a
 a
.. _CRFEntityExtractor: a
 a
CRFEntityExtractor a
~~~~~~~~~~~~~~~~~~ a
 a
:Short: Conditional random field (CRF) entity extraction a
:Outputs: ``entities`` a
:Requires: ``tokens`` and ``dense_features`` (optional) a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "entities": [{ a
                "value": "New York City", a
                "start": 20, a
                "end": 33, a
                "entity": "city", a
                "confidence": 0.874, a
                "extractor": "CRFEntityExtractor" a
            }] a
        } a
 a
:Description: a
    This component implements a conditional random fields (CRF) to do named entity recognition. a
    CRFs can be thought of as an undirected Markov chain where the time steps are words a
    and the states are entity classes. Features of the words (capitalisation, POS tagging, a
    etc.) give probabilities to certain entity classes, as are transitions between a
    neighbouring entity tags: the most likely set of tags is then calculated and returned. a
 a
:Configuration: a
    ``CRFEntityExtractor`` has a list of default features to use. a
    However, you can overwrite the default configuration. a
    The following features are available: a
 a
    .. code-block:: none a
 a
        ==============  ========================================================================================== a
        Feature Name    Description a
        ==============  ========================================================================================== a
        low             Checks if the token is lower case. a
        upper           Checks if the token is upper case. a
        title           Checks if the token starts with an uppercase character and all remaining characters are a
                        lowercased. a
        digit           Checks if the token contains just digits. a
        prefix5         Take the first five characters of the token. a
        prefix2         Take the first two characters of the token. a
        suffix5         Take the last five characters of the token. a
        suffix3         Take the last three characters of the token. a
        suffix2         Take the last two characters of the token. a
        suffix1         Take the last character of the token. a
        pos             Take the Part-of-Speech tag of the token (``SpacyTokenizer`` required). a
        pos2            Take the first two characters of the Part-of-Speech tag of the token a
                        (``SpacyTokenizer`` required). a
        pattern         Take the patterns defined by ``RegexFeaturizer``. a
        bias            Add an additional "bias" feature to the list of features. a
        ==============  ========================================================================================== a
 a
    As the featurizer is moving over the tokens in a user message with a sliding window, you can define features for a
    previous tokens, the current token, and the next tokens in the sliding window. a
    You define the features as [before, token, after] array. a
 a
    Additional you can set a flag to determine whether to use the BILOU tagging schema or not. a
 a
        - ``BILOU_flag`` determines whether to use BILOU tagging or not. Default ``True``. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "CRFEntityExtractor" a
          # BILOU_flag determines whether to use BILOU tagging or not. a
          "BILOU_flag": True a
          # features to extract in the sliding window a
          "features": [ a
            ["low", "title", "upper"], a
            [ a
              "bias", a
              "low", a
              "prefix5", a
              "prefix2", a
              "suffix5", a
              "suffix3", a
              "suffix2", a
              "upper", a
              "title", a
              "digit", a
              "pattern", a
            ], a
            ["low", "title", "upper"], a
          ] a
          # The maximum number of iterations for optimization algorithms. a
          "max_iterations": 50 a
          # weight of the L1 regularization a
          "L1_c": 0.1 a
          # weight of the L2 regularization a
          "L2_c": 0.1 a
 a
    .. note:: a
        If POS features are used (``pos`` or ``pos2`), you need to have ``SpacyTokenizer`` in your pipeline. a
 a
    .. note:: a
        If "``pattern` features are used, you need to have ``RegexFeaturizer`` in your pipeline. a
 a
.. _DucklingHTTPExtractor: a
 a
DucklingHTTPExtractor a
~~~~~~~~~~~~~~~~~~~~~ a
 a
:Short: Duckling lets you extract common entities like dates, a
        amounts of money, distances, and others in a number of languages. a
:Outputs: ``entities`` a
:Requires: Nothing a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "entities": [{ a
                "end": 53, a
                "entity": "time", a
                "start": 48, a
                "value": "2017-04-10T00:00:00.000+02:00", a
                "confidence": 1.0, a
                "extractor": "DucklingHTTPExtractor" a
            }] a
        } a
 a
:Description: a
    To use this component you need to run a duckling server. The easiest a
    option is to spin up a docker container using a
    ``docker run -p 8000:8000 rasa/duckling``. a
 a
    Alternatively, you can `install duckling directly on your a
    machine <https://github.com/facebook/duckling#quickstart>`_ and start the server. a
 a
    Duckling allows to recognize dates, numbers, distances and other structured entities a
    and normalizes them. a
    Please be aware that duckling tries to extract as many entity types as possible without a
    providing a ranking. For example, if you specify both ``number`` and ``time`` as dimensions a
    for the duckling component, the component will extract two entities: ``10`` as a number and a
    ``in 10 minutes`` as a time from the text ``I will be there in 10 minutes``. In such a a
    situation, your application would have to decide which entity type is be the correct one. a
    The extractor will always return `1.0` as a confidence, as it is a rule a
    based system. a
 a
:Configuration: a
    Configure which dimensions, i.e. entity types, the duckling component a
    should extract. A full list of available dimensions can be found in a
    the `duckling documentation <https://duckling.wit.ai/>`_. a
    Leaving the dimensions option unspecified will extract all available dimensions. a
 a
    .. code-block:: yaml a
 a
        pipeline: a
        - name: "DucklingHTTPExtractor" a
          # url of the running duckling server a
          url: "http://localhost:8000" a
          # dimensions to extract a
          dimensions: ["time", "number", "amount-of-money", "distance"] a
          # allows you to configure the locale, by default the language is a
          # used a
          locale: "de_DE" a
          # if not set the default timezone of Duckling is going to be used a
          # needed to calculate dates from relative expressions like "tomorrow" a
          timezone: "Europe/Berlin" a
          # Timeout for receiving response from http url of the running duckling server a
          # if not set the default timeout of duckling http url is set to 3 seconds. a
          timeout : 3 a
 a
DIETClassifier a
~~~~~~~~~~~~~~ a
 a
:Short: Dual Intent Entity Transformer (DIET) used for intent classification and entity extraction a
:Description: a
    You can find the detailed description of the :ref:`diet-classifier` under the section a
    `Combined Entity Extractors and Intent Classifiers`. a
 a
 a
Selectors a
---------- a
 a
Selectors predict a bot response from a set of candidate responses. a
 a
.. _response-selector: a
 a
ResponseSelector a
~~~~~~~~~~~~~~~~ a
 a
:Short: Response Selector a
:Outputs: A dictionary with key as ``direct_response_intent`` and value containing ``response`` and ``ranking`` a
:Requires: ``dense_features`` and/or ``sparse_features`` for user messages and response a
 a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "response_selector": { a
              "faq": { a
                "response": {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"}, a
                "ranking": [ a
                    {"confidence": 0.7356462617, "name": "Supports 3.5, 3.6 and 3.7, recommended version is 3.6"}, a
                    {"confidence": 0.2134543431, "name": "You can ask me about how to get started"} a
                ] a
              } a
            } a
        } a
 a
:Description: a
 a
    Response Selector component can be used to build a response retrieval model to directly predict a bot response from a
    a set of candidate responses. The prediction of this model is used by :ref:`retrieval-actions`. a
    It embeds user inputs and response labels into the same space and follows the exact same a
    neural network architecture and optimization as the :ref:`diet-classifier`. a
 a
    .. note:: If during prediction time a message contains **only** words unseen during training a
              and no Out-Of-Vocabulary preprocessor was used, an empty response ``None`` is predicted with confidence a
              ``0.0``. This might happen if you only use the :ref:`CountVectorsFeaturizer` with a ``word`` analyzer a
              as featurizer. If you use the ``char_wb`` analyzer, you should always get a response with a confidence a
              value ``> 0.0``. a
 a
:Configuration: a
 a
    The algorithm includes almost all the hyperparameters that :ref:`diet-classifier` uses. a
    If you want to adapt your model, start by modifying the following parameters: a
 a
        - ``epochs``: a
          This parameter sets the number of times the algorithm will see the training data (default: ``300``). a
          One ``epoch`` is equals to one forward pass and one backward pass of all the training examples. a
          Sometimes the model needs more epochs to properly learn. a
          Sometimes more epochs don't influence the performance. a
          The lower the number of epochs the faster the model is trained. a
        - ``hidden_layers_sizes``: a
          This parameter allows you to define the number of feed forward layers and their output a
          dimensions for user messages and intents (default: ``text: [256, 128], label: [256, 128]``). a
          Every entry in the list corresponds to a feed forward layer. a
          For example, if you set ``text: [256, 128]``, we will add two feed forward layers in front of a
          the transformer. The vectors of the input tokens (coming from the user message) will be passed on to those a
          layers. The first layer will have an output dimension of 256 and the second layer will have an output a
          dimension of 128. If an empty list is used (default behaviour), no feed forward layer will be a
          added. a
          Make sure to use only positive integer values. Usually, numbers of power of two are used. a
          Also, it is usual practice to have decreasing values in the list: next value is smaller or equal to the a
          value before. a
        - ``embedding_dimension``: a
          This parameter defines the output dimension of the embedding layers used inside the model (default: ``20``). a
          We are using multiple embeddings layers inside the model architecture. a
          For example, the vector of the ``__CLS__`` token and the intent is passed on to an embedding layer before a
          they are compared and the loss is calculated. a
        - ``number_of_transformer_layers``: a
          This parameter sets the number of transformer layers to use (default: ``0``). a
          The number of transformer layers corresponds to the transformer blocks to use for the model. a
        - ``transformer_size``: a
          This parameter sets the number of units in the transformer (default: ``None``). a
          The vectors coming out of the transformers will have the given ``transformer_size``. a
        - ``weight_sparsity``: a
          This parameter defines the fraction of kernel weights that are set to 0 for all feed forward layers a
          in the model (default: ``0.8``). The value should be between 0 and 1. If you set ``weight_sparsity`` a
          to 0, no kernel weights will be set to 0, the layer acts as a standard feed forward layer. You should not a
          set ``weight_sparsity`` to 1 as this would result in all kernel weights being 0, i.e. the model is not able a
          to learn. a
 a
    | a
 a
    In addition, the component can also be configured to train a response selector for a particular retrieval intent. a
    The parameter ``retrieval_intent`` sets the name of the intent for which this response selector model is trained. a
    Default is ``None``, i.e. the model is trained for all retrieval intents. a
 a
    | a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            The above configuration parameters are the ones you should configure to fit your model to your data. a
            However, additional parameters exist that can be adapted. a
 a
        .. code-block:: none a
 a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | Parameter                       | Default Value     | Description                                                  | a
         +=================================+===================+==============================================================+ a
         | hidden_layers_sizes             | text: [256, 128]  | Hidden layer sizes for layers before the embedding layers    | a
         |                                 | label: [256, 128] | for user messages and labels. The number of hidden layers is | a
         |                                 |                   | equal to the length of the corresponding.                    | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | share_hidden_layers             | False             | Whether to share the hidden layer weights between user       | a
         |                                 |                   | messages and labels.                                         | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | transformer_size                | None              | Number of units in transformer.                              | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | number_of_transformer_layers    | 0                 | Number of transformer layers.                                | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | number_of_attention_heads       | 4                 | Number of attention heads in transformer.                    | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | use_key_relative_attention      | False             | If 'True' use key relative embeddings in attention.          | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | use_value_relative_attention    | False             | If 'True' use value relative embeddings in attention.        | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | max_relative_position           | None              | Maximum position for relative embeddings.                    | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | unidirectional_encoder          | False             | Use a unidirectional or bidirectional encoder.               | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | batch_size                      | [64, 256]         | Initial and final value for batch sizes.                     | a
         |                                 |                   | Batch size will be linearly increased for each epoch.        | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | batch_strategy                  | "balanced"        | Strategy used when creating batches.                         | a
         |                                 |                   | Can be either 'sequence' or 'balanced'.                      | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | epochs                          | 300               | Number of epochs to train.                                   | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | random_seed                     | None              | Set random seed to any 'int' to get reproducible results.    | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | learning_rate                   | 0.001             | Initial learning rate for the optimizer.                     | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | embedding_dimension             | 20                | Dimension size of embedding vectors.                         | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | dense_dimension                 | text: 512         | Dense dimension for sparse features to use if no dense       | a
         |                                 | label: 512        | features are present.                                        | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | number_of_negative_examples     | 20                | The number of incorrect labels. The algorithm will minimize  | a
         |                                 |                   | their similarity to the user input during training.          | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | similarity_type                 | "auto"            | Type of similarity measure to use, either 'auto' or 'cosine' | a
         |                                 |                   | or 'inner'.                                                  | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | loss_type                       | "softmax"         | The type of the loss function, either 'softmax' or 'margin'. | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | ranking_length                  | 10                | Number of top actions to normalize scores for loss type      | a
         |                                 |                   | 'softmax'. Set to 0 to turn off normalization.               | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | maximum_positive_similarity     | 0.8               | Indicates how similar the algorithm should try to make       | a
         |                                 |                   | embedding vectors for correct labels.                        | a
         |                                 |                   | Should be 0.0 < ... < 1.0 for 'cosine' similarity type.      | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | maximum_negative_similarity     | -0.4              | Maximum negative similarity for incorrect labels.            | a
         |                                 |                   | Should be -1.0 < ... < 1.0 for 'cosine' similarity type.     | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | use_maximum_negative_similarity | True              | If 'True' the algorithm only minimizes maximum similarity    | a
         |                                 |                   | over incorrect intent labels, used only if 'loss_type' is    | a
         |                                 |                   | set to 'margin'.                                             | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | scale_loss                      | True              | Scale loss inverse proportionally to confidence of correct   | a
         |                                 |                   | prediction.                                                  | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | regularization_constant         | 0.002             | The scale of regularization.                                 | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | negative_margin_scale           | 0.8               | The scale of how important is to minimize the maximum        | a
         |                                 |                   | similarity between embeddings of different labels.           | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | weight_sparsity                 | 0.8               | Sparsity of the weights in dense layers.                     | a
         |                                 |                   | Value should be between 0 and 1.                             | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | drop_rate                       | 0.2               | Dropout rate for encoder. Value should be between 0 and 1.   | a
         |                                 |                   | The higher the value the higher the regularization effect.   | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | drop_rate_attention             | 0.0               | Dropout rate for attention. Value should be between 0 and 1. | a
         |                                 |                   | The higher the value the higher the regularization effect.   | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | use_sparse_input_dropout        | False             | If 'True' apply dropout to sparse input tensors.             | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | use_dense_input_dropout         | False             | If 'True' apply dropout to dense input tensors.              | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | evaluate_every_number_of_epochs | 20                | How often to calculate validation accuracy.                  | a
         |                                 |                   | Set to '-1' to evaluate just once at the end of training.    | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | evaluate_on_number_of_examples  | 0                 | How many examples to use for hold out validation set.        | a
         |                                 |                   | Large values may hurt performance, e.g. model accuracy.      | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | use_masked_language_model       | False             | If 'True' random tokens of the input message will be masked  | a
         |                                 |                   | and the model should predict those tokens.                   | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | retrieval_intent                | None              | Name of the intent for which this response selector model is | a
         |                                 |                   | trained.                                                     | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | tensorboard_log_directory       | None              | If you want to use tensorboard to visualize training         | a
         |                                 |                   | metrics, set this option to a valid output directory. You    | a
         |                                 |                   | can view the training metrics after training in tensorboard  | a
         |                                 |                   | via 'tensorboard --logdir <path-to-given-directory>'.        | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
         | tensorboard_log_level           | "epoch"           | Define when training metrics for tensorboard should be       | a
         |                                 |                   | logged. Either after every epoch ("epoch") or for every      | a
         |                                 |                   | training step ("minibatch").                                 | a
         +---------------------------------+-------------------+--------------------------------------------------------------+ a
 a
        .. note:: For ``cosine`` similarity ``maximum_positive_similarity`` and ``maximum_negative_similarity`` should a
                  be between ``-1`` and ``1``. a
 a
        .. note:: There is an option to use linearly increasing batch size. The idea comes from a
                  `<https://arxiv.org/abs/1711.00489>`_. a
                  In order to do it pass a list to ``batch_size``, e.g. ``"batch_size": [64, 256]`` (default behaviour). a
                  If constant ``batch_size`` is required, pass an ``int``, e.g. ``"batch_size": 64``. a
 a
        .. note:: Parameter ``maximum_negative_similarity`` is set to a negative value to mimic the original a
                  starspace algorithm in the case ``maximum_negative_similarity = maximum_positive_similarity`` a
                  and ``use_maximum_negative_similarity = False``. a
                  See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details. a
 a
 a
Combined Entity Extractors and Intent Classifiers a
------------------------------------------------- a
 a
.. _diet-classifier: a
 a
DIETClassifier a
~~~~~~~~~~~~~~ a
 a
:Short: Dual Intent Entity Transformer (DIET) used for intent classification and entity extraction a
:Outputs: ``entities``, ``intent`` and ``intent_ranking`` a
:Requires: ``dense_features`` and/or ``sparse_features`` for user message and optionally the intent a
:Output-Example: a
 a
    .. code-block:: json a
 a
        { a
            "intent": {"name": "greet", "confidence": 0.8343}, a
            "intent_ranking": [ a
                { a
                    "confidence": 0.385910906220309, a
                    "name": "goodbye" a
                }, a
                { a
                    "confidence": 0.28161531595656784, a
                    "name": "restaurant_search" a
                } a
            ], a
            "entities": [{ a
                "end": 53, a
                "entity": "time", a
                "start": 48, a
                "value": "2017-04-10T00:00:00.000+02:00", a
                "confidence": 1.0, a
                "extractor": "DIETClassifier" a
            }] a
        } a
 a
:Description: a
    DIET (Dual Intent and Entity Transformer) is a multi-task architecture for intent classification and entity a
    recognition. The architecture is based on a transformer which is shared for both tasks. a
    A sequence of entity labels is predicted through a Conditional Random Field (CRF) tagging layer on top of the a
    transformer output sequence corresponding to the input sequence of tokens. a
    For the intent labels the transformer output for the ``__CLS__`` token and intent labels are embedded into a a
    single semantic vector space. We use the dot-product loss to maximize the similarity with the target label and a
    minimize similarities with negative samples. a
 a
    If you want to learn more about the model, please take a look at our a
    `videos <https://www.youtube.com/playlist?list=PL75e0qA87dlG-za8eLI6t0_Pbxafk-cxb>`__ where we explain the model a
    architecture in detail. a
 a
    .. note:: If during prediction time a message contains **only** words unseen during training a
              and no Out-Of-Vocabulary preprocessor was used, an empty intent ``None`` is predicted with confidence a
              ``0.0``. This might happen if you only use the :ref:`CountVectorsFeaturizer` with a ``word`` analyzer a
              as featurizer. If you use the ``char_wb`` analyzer, you should always get an intent with a confidence a
              value ``> 0.0``. a
 a
:Configuration: a
 a
    If you want to use the ``DIETClassifier`` just for intent classification, set ``entity_recognition`` to ``False``. a
    If you want to do only entity recognition, set ``intent_classification`` to ``False``. a
    By default ``DIETClassifier`` does both, i.e. ``entity_recognition`` and ``intent_classification`` are set to a
    ``True``. a
 a
    You can define a number of hyperparameters to adapt the model. a
    If you want to adapt your model, start by modifying the following parameters: a
 a
        - ``epochs``: a
          This parameter sets the number of times the algorithm will see the training data (default: ``300``). a
          One ``epoch`` is equals to one forward pass and one backward pass of all the training examples. a
          Sometimes the model needs more epochs to properly learn. a
          Sometimes more epochs don't influence the performance. a
          The lower the number of epochs the faster the model is trained. a
        - ``hidden_layers_sizes``: a
          This parameter allows you to define the number of feed forward layers and their output a
          dimensions for user messages and intents (default: ``text: [], label: []``). a
          Every entry in the list corresponds to a feed forward layer. a
          For example, if you set ``text: [256, 128]``, we will add two feed forward layers in front of a
          the transformer. The vectors of the input tokens (coming from the user message) will be passed on to those a
          layers. The first layer will have an output dimension of 256 and the second layer will have an output a
          dimension of 128. If an empty list is used (default behaviour), no feed forward layer will be a
          added. a
          Make sure to use only positive integer values. Usually, numbers of power of two are used. a
          Also, it is usual practice to have decreasing values in the list: next value is smaller or equal to the a
          value before. a
        - ``embedding_dimension``: a
          This parameter defines the output dimension of the embedding layers used inside the model (default: ``20``). a
          We are using multiple embeddings layers inside the model architecture. a
          For example, the vector of the ``__CLS__`` token and the intent is passed on to an embedding layer before a
          they are compared and the loss is calculated. a
        - ``number_of_transformer_layers``: a
          This parameter sets the number of transformer layers to use (default: ``2``). a
          The number of transformer layers corresponds to the transformer blocks to use for the model. a
        - ``transformer_size``: a
          This parameter sets the number of units in the transformer (default: ``256``). a
          The vectors coming out of the transformers will have the given ``transformer_size``. a
        - ``weight_sparsity``: a
          This parameter defines the fraction of kernel weights that are set to 0 for all feed forward layers a
          in the model (default: ``0.8``). The value should be between 0 and 1. If you set ``weight_sparsity`` a
          to 0, no kernel weights will be set to 0, the layer acts as a standard feed forward layer. You should not a
          set ``weight_sparsity`` to 1 as this would result in all kernel weights being 0, i.e. the model is not able a
          to learn. a
 a
    .. container:: toggle a
 a
        .. container:: header a
 a
            The above configuration parameters are the ones you should configure to fit your model to your data. a
            However, additional parameters exist that can be adapted. a
 a
        .. code-block:: none a
 a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | Parameter                       | Default Value    | Description                                                  | a
         +=================================+==================+==============================================================+ a
         | hidden_layers_sizes             | text: []         | Hidden layer sizes for layers before the embedding layers    | a
         |                                 | label: []        | for user messages and labels. The number of hidden layers is | a
         |                                 |                  | equal to the length of the corresponding.                    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | share_hidden_layers             | False            | Whether to share the hidden layer weights between user       | a
         |                                 |                  | messages and labels.                                         | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | transformer_size                | 256              | Number of units in transformer.                              | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | number_of_transformer_layers    | 2                | Number of transformer layers.                                | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | number_of_attention_heads       | 4                | Number of attention heads in transformer.                    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_key_relative_attention      | False            | If 'True' use key relative embeddings in attention.          | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_value_relative_attention    | False            | If 'True' use value relative embeddings in attention.        | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | max_relative_position           | None             | Maximum position for relative embeddings.                    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | unidirectional_encoder          | False            | Use a unidirectional or bidirectional encoder.               | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | batch_size                      | [64, 256]        | Initial and final value for batch sizes.                     | a
         |                                 |                  | Batch size will be linearly increased for each epoch.        | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | batch_strategy                  | "balanced"       | Strategy used when creating batches.                         | a
         |                                 |                  | Can be either 'sequence' or 'balanced'.                      | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | epochs                          | 300              | Number of epochs to train.                                   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | random_seed                     | None             | Set random seed to any 'int' to get reproducible results.    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | learning_rate                   | 0.001            | Initial learning rate for the optimizer.                     | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | embedding_dimension             | 20               | Dimension size of embedding vectors.                         | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | dense_dimension                 | text: 512        | Dense dimension for sparse features to use if no dense       | a
         |                                 | label: 20        | features are present.                                        | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | number_of_negative_examples     | 20               | The number of incorrect labels. The algorithm will minimize  | a
         |                                 |                  | their similarity to the user input during training.          | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | similarity_type                 | "auto"           | Type of similarity measure to use, either 'auto' or 'cosine' | a
         |                                 |                  | or 'inner'.                                                  | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | loss_type                       | "softmax"        | The type of the loss function, either 'softmax' or 'margin'. | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | ranking_length                  | 10               | Number of top actions to normalize scores for loss type      | a
         |                                 |                  | 'softmax'. Set to 0 to turn off normalization.               | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | maximum_positive_similarity     | 0.8              | Indicates how similar the algorithm should try to make       | a
         |                                 |                  | embedding vectors for correct labels.                        | a
         |                                 |                  | Should be 0.0 < ... < 1.0 for 'cosine' similarity type.      | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | maximum_negative_similarity     | -0.4             | Maximum negative similarity for incorrect labels.            | a
         |                                 |                  | Should be -1.0 < ... < 1.0 for 'cosine' similarity type.     | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_maximum_negative_similarity | True             | If 'True' the algorithm only minimizes maximum similarity    | a
         |                                 |                  | over incorrect intent labels, used only if 'loss_type' is    | a
         |                                 |                  | set to 'margin'.                                             | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | scale_loss                      | False            | Scale loss inverse proportionally to confidence of correct   | a
         |                                 |                  | prediction.                                                  | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | regularization_constant         | 0.002            | The scale of regularization.                                 | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | negative_margin_scale           | 0.8              | The scale of how important it is to minimize the maximum     | a
         |                                 |                  | similarity between embeddings of different labels.           | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | weight_sparsity                 | 0.8              | Sparsity of the weights in dense layers.                     | a
         |                                 |                  | Value should be between 0 and 1.                             | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | drop_rate                       | 0.2              | Dropout rate for encoder. Value should be between 0 and 1.   | a
         |                                 |                  | The higher the value the higher the regularization effect.   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | drop_rate_attention             | 0.0              | Dropout rate for attention. Value should be between 0 and 1. | a
         |                                 |                  | The higher the value the higher the regularization effect.   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_sparse_input_dropout        | True             | If 'True' apply dropout to sparse input tensors.             | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_dense_input_dropout         | True             | If 'True' apply dropout to dense input tensors.              | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | evaluate_every_number_of_epochs | 20               | How often to calculate validation accuracy.                  | a
         |                                 |                  | Set to '-1' to evaluate just once at the end of training.    | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | evaluate_on_number_of_examples  | 0                | How many examples to use for hold out validation set.        | a
         |                                 |                  | Large values may hurt performance, e.g. model accuracy.      | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | intent_classification           | True             | If 'True' intent classification is trained and intents are   | a
         |                                 |                  | predicted.                                                   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | entity_recognition              | True             | If 'True' entity recognition is trained and entities are     | a
         |                                 |                  | extracted.                                                   | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | use_masked_language_model       | False            | If 'True' random tokens of the input message will be masked  | a
         |                                 |                  | and the model has to predict those tokens. It acts like a    | a
         |                                 |                  | regularizer and should help to learn a better contextual     | a
         |                                 |                  | representation of the input.                                 | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | tensorboard_log_directory       | None             | If you want to use tensorboard to visualize training         | a
         |                                 |                  | metrics, set this option to a valid output directory. You    | a
         |                                 |                  | can view the training metrics after training in tensorboard  | a
         |                                 |                  | via 'tensorboard --logdir <path-to-given-directory>'.        | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
         | tensorboard_log_level           | "epoch"          | Define when training metrics for tensorboard should be       | a
         |                                 |                  | logged. Either after every epoch ('epoch') or for every      | a
         |                                 |                  | training step ('minibatch').                                 | a
         +---------------------------------+------------------+--------------------------------------------------------------+ a
 a
        .. note:: For ``cosine`` similarity ``maximum_positive_similarity`` and ``maximum_negative_similarity`` should a
                  be between ``-1`` and ``1``. a
 a
        .. note:: There is an option to use linearly increasing batch size. The idea comes from a
                  `<https://arxiv.org/abs/1711.00489>`_. a
                  In order to do it pass a list to ``batch_size``, e.g. ``"batch_size": [64, 256]`` (default behaviour). a
                  If constant ``batch_size`` is required, pass an ``int``, e.g. ``"batch_size": 64``. a
 a
        .. note:: Parameter ``maximum_negative_similarity`` is set to a negative value to mimic the original a
                  starspace algorithm in the case ``maximum_negative_similarity = maximum_positive_similarity`` a
                  and ``use_maximum_negative_similarity = False``. a
                  See `starspace paper <https://arxiv.org/abs/1709.03856>`_ for details. a
 a