.. _section_configuration:

Configuration
=============

You can provide options to rasa NLU through:

- a json-formatted config file
- environment variables
- command line arguments

Environment variables override options in your config file, 
and command line args will override any options specified elsewhere.
Environment variables are capitalised and prefixed with ``RASA_``, 
so the option ``pipeline`` is specified with the ``RASA_PIPELINE`` env var.

Default
-------
Here is the default configuration including all available parameters:

.. literalinclude:: ../sample_configs/config_defaults.json
    :language: json

Options
-------
A short explanation and examples for each configuration value.

project
~~~~~~~

:Type: ``str``
:Examples: ``"my_project_name"``
:Description:
     Defines a project name to train new models for and to refer to when using the http server.
     The default value is ``null`` which will lead to using the default project ``"default"``.
     All projects are stored under the ``path`` directory.

pipeline
~~~~~~~~

:Type: ``str`` or ``[str]``
:Examples:
    ``"mitie"`` or
    ``["nlp_spacy", "ner_spacy", "ner_synonyms"]``

:Description:
    The pipeline used for training. Can either be a template (passing a string) or a list of components (array). For all
    available templates, see :ref:`section_pipeline`.

language
~~~~~~~~

:Type: ``str``
:Examples: ``"en"`` or ``"de"``
:Description:
    Language the model is trained in. Underlying word vectors will be loaded by using this language

num_threads
~~~~~~~~~~~

:Type: ``int``
:Examples: ``4``
:Description:
    Number of threads used during training (not supported by all components, though.
    Some of them might still be single threaded!).

fixed_model_name
~~~~~~~~~~~~~~~~

:Type: ``str``
:Examples: ``"my_model_name"``
:Description:
    Instead of generating model names (e.g. ``model_20170922-234435``) a fixed
    model name will be used. The model will always be saved in the path
    ``{project_path}/{project_name}/{model_name}``. If the model is assigned
    a fixed name, it will possibly override previously trained models.

max_training_processes
~~~~~~~~~~~

:Type: ``int``
:Examples: ``1``
:Description:
    Number of processes used to handle training requests. Increasing this value will have a great impact on memory usage.
    It is recommended to keep the default value.

path
~~~~

:Type: ``str``
:Examples: ``"projects/"``
:Description:
    Projects directory where trained models will be saved to (training) and
    loaded from (http server).

response_log
~~~~~~~~~~~~

:Type: ``str`` or ``null``
:Examples: ``"logs/"``
:Description:
    Directory where logs will be saved (containing queries and responses).
    If set to ``null`` logging will be disabled.

config
~~~~~~

:Type: ``str``
:Examples: ``"sample_configs/config_spacy.json"``
:Description:
    Location of the configuration file (can only be set as env var or command line option).

log_level
~~~~~~~~~

:Type: ``str``
:Examples: ``"DEBUG"``
:Description:
    Log level used to output messages from the framework internals.

port
~~~~

:Type: ``int``
:Examples: ``5000``
:Description:
    Port on which to run the http server.

data
~~~~

:Type: ``str``
:Examples: ``"data/example.json"``
:Description:
    Location of the training data. For JSON and markdown data, this can either be a single file or a directory containing multiple training data files.

cors_origins
~~~~

:Type: ``list``
:Examples: ``["*"]``, ``["*.mydomain.com", "api.domain2.net"]``
:Description:
    List of domain patterns from where CORS (cross-origin resource sharing) calls are allowed.
    The default value is ``[]`` which forbids all CORS requests.

emulate
~~~~~~~

:Type: ``str``
:Examples: ``"wit"``, ``"luis"`` or ``"api"``
:Description:
    Format to be returned by the http server. If ``null`` (default) the rasa NLU internal format will be used.
    Otherwise, the output will be formatted according to the API specified.

mitie_file
~~~~~~~~~~

:Type: ``str``
:Examples: ``"data/total_word_feature_extractor.dat"``
:Description:
    File containing ``total_word_feature_extractor.dat`` (see :ref:`section_backends`)

spacy_model_name
~~~~~~~~~~~~~~~~

:Type: ``str``
:Examples: ``"en_core_web_md"``
:Description:
    If the spacy model to be used has a name that is different from the language tag (``"en"``, ``"de"``, etc.),
    the model name can be specified using this configuration variable. The name will be passed to ``spacy.load(name)``.

token
~~~~~

:Type: ``str`` or ``null``
:Examples: ``"asd2aw3r"``
:Description:
    if set, all requests to server must have a ``?token=<token>`` query param. see :ref:`section_auth`

max_number_of_ngrams
~~~~~~~~~~~~~~~~~~~~

:Type: ``int``
:Examples: ``10``
:Description:
    Maximum number of ngrams to use when augmenting feature vectors with character ngrams
    (``intent_featurizer_ngrams`` component only)

.. _section_configuration_duckling_dimensions:

duckling_dimensions
~~~~~~~~~~~~~~~~~~~

:Type: ``list``
:Examples: ``["time", "number", "amount-of-money", "distance"]``
:Description:
    Defines which dimensions, i.e. entity types, the :ref:`duckling component <section_pipeline_duckling>` will extract.
    A full list of available dimensions can be found in the `duckling documentation <https://duckling.wit.ai/>`_.

storage
~~~~~~~

:Type: ``str``
:Examples: ``"aws"`` or ``"gcs"``
:Description:
    Storage type for persistor. See :ref:`section_persistence` for more details.

bucket_name
~~~~~~~~~~~

:Type: ``str``
:Examples: ``"my_models"``
:Description:
    Name of the bucket in the cloud to store the models. If the specified bucket name does not exist, rasa will create it.
    See :ref:`section_persistence` for more details.

aws_region
~~~~~~~~~~

:Type: ``str``
:Examples: ``"us-east-1"``
:Description:
    Name of the aws region to use. This is used only when ``"storage"`` is selected as ``"aws"``.
    See :ref:`section_persistence` for more details.

aws_endpoint_url
~~~~~~~~~~

:Type: ``str``
:Examples: ``"http://10.0.0.1:9000"``
:Description:
    Optional endpoint of the custom S3 compatible storage provider. This is used only when ``"storage"`` is selected as ``"aws"``.
    See :ref:`section_persistence` for more details.

ner_crf
~~~~~~~

features
++++++++

:Type: ``[[str]]``
:Examples: ``[["low", "title"], ["bias", "word3"], ["upper", "pos", "pos2"]]``
:Description:
    The features are a ``[before, word, after]`` array with before, word, after holding keys about which
    features to use for each word, for example, ``"title"`` in array before will have the feature
    "is the preceding word in title case?".
    Available features are:
    ``low``, ``title``, ``word3``, ``word2``, ``pos``, ``pos2``, ``bias``, ``upper`` and ``digit``

BILOU_flag
++++++++++

:Type: ``bool``
:Examples: ``true``
:Description:
     The flag determines whether to use BILOU tagging or not. BILOU tagging is more rigorous however
     requires more examples per entity. Rule of thumb: use only if more than 100 examples per entity.

max_iterations
++++++++++++++

:Type: ``int``
:Examples: ``50``
:Description:
    This is the value given to sklearn_crfcuite.CRF tagger before training.

L1_C
++++

:Type: ``float``
:Examples: ``1.0``
:Description:
    This is the value given to sklearn_crfcuite.CRF tagger before training.
    Specifies the L1 regularization coefficient.

L2_C
++++

:Type: ``float``
:Examples: ``1e-3``
:Description:
    This is the value given to sklearn_crfcuite.CRF tagger before training.
    Specifies the L2 regularization coefficient.

intent_classifier_sklearn
~~~~~~~~~~~~~~~~~~~~~~~~~

C
+

:Type: ``[float]``
:Examples: ``[1, 2, 5, 10, 20, 100]``
:Description:
    Specifies the list of regularization values to cross-validate over for C-SVM.
    This is used with the ``kernel`` hyperparameter in GridSearchCV.

kernel
++++++

:Type: ``string``
:Examples: ``"linear"``
:Description:
    Specifies the kernel to use with C-SVM.
    This is used with the ``C`` hyperparameter in GridSearchCV.
