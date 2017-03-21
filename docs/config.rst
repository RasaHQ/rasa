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

.. literalinclude:: ../config_defaults.json
    :language: json

Options
-------
A short explanation and examples for each configuration value.

pipeline
~~~~~~~~

:Type: ``str`` or ``[str]``
:Examples:
    ``"mitie"`` or
    ``["init_spacy", "ner_spacy", "ner_synonyms"]``

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

path
~~~~

:Type: ``str``
:Examples: ``"models/"``
:Description:
    Directory where trained models will be saved to (training) and loaded from (http server).

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
:Examples: ``"config_spacy.json"``
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
    Location of the training data.

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

fine_tune_spacy_ner
~~~~~~~~~~~~~~~~~~~

:Type: ``bool``
:Examples: ``true``
:Description:
    Fine tune existing spacy NER models vs training from scratch. (``ner_spacy`` component only)

server_model_dirs
~~~~~~~~~~~~~~~~~

:Type: ``str``
:Examples: ``models/``
:Description:
    Directory containing the model to be used by server or an object describing multiple models.
    see :ref:`HTTP server config<section_http_config>`

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

luis_data_tokenizer
~~~~~~~~~~~~~~~~~~~

:Type: ``str``
:Examples: ``"tokenizer_mitie"``
:Description:
    Name of the tokenization component used to process luis data (Luis data annotates entities using token offset
    instead of character offsets, to convert the token offsets to character positions a tokenizer is required.)
    see :ref:`section_migration`

If you want to persist your trained models to S3, there are additional configuration options,
see :ref:`section_persistence`
