Change Log
==========

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 0.7.0.

[Unreleased] - `master`_
^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: This version is not yet released and is under active development.

Added
-----
- Support for training data in Markdown format
- Cors support. You can now specify allowed cors origins within your configuration file.
- The HTTP server is now backed by Klein (Twisted) instead of Flask. The server is now asynchronous but is no more WSGI compatible
- Improved Docker automated builds

Changed
-------
- Root project directories have been slightly rearranged to clean up new docker support
- use `Interpreter.create(metadata, ...)` to create interpreter from dict and `Interpreter.load(file_name, ...)` to create interpreter with metadata from a file

Fixed
-----
- Fixed training data persistence. Fixes #510
- Fixed UTF-8 character handling when training through HTTP interface
- Invalid handling of numbers extracted from duckling during synonym handling. Fixes #517
- Only log a warning (instead of throwing an exception) on misaligned entities during mitie NER

[0.9.0] - 2017-07-07
^^^^^^^^^^^^^^^^^^^^

Added
-----
- increased test coverage to avoid regressions (ongoing)
- added regex featurization to support intent classification and entity extraction (``intent_entity_featurizer_regex``)

Changed
-------
- replaced existing CRF library (python-crfsuite) with sklearn-crfsuite (due to better windows support)
- updated to spacy 1.8.2
- logging format of logged request now includes model name and timestamp
- use module specific loggers instead of default python root logger
- output format of the duckling extractor changed. the ``value`` field now includes the complete value from duckling instead of just text (so this is an property is an object now instead of just text). includes granularity information now.
- deprecated ``intent_examples`` and ``entity_examples`` sections in training data. all examples should go into the ``common_examples`` section
- weight training samples based on class distribution during ner_crf cross validation and sklearn intent classification training
- large refactoring of the internal training data structure and pipeline architecture
- numpy is now a required dependency

Removed
-------
- luis data tokenizer configuration value (not used anymore, luis exports char offsets now)

Fixed
-----
- properly update coveralls coverage report from travis
- persistence of duckling dimensions
- changed default response of untrained ``intent_classifier_sklearn`` from ``"intent": None`` to ``"intent": {"name": None, "confidence": 0.0}``
- ``/status`` endpoint showing all available models instead of only those whose name starts with *model*
- properly return training process ids #391

[0.8.12] - 2017-06-29
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed missing argument attribute error


[0.8.11] - 2017-06-07
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- updated mitie installation documentation

[0.8.10] - 2017-05-31
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed documentation about training data format


[0.8.9] - 2017-05-26
^^^^^^^^^^^^^^^^^^^^

Fixed
-----^
- properly handle response_log configuration variable being set to ``null``

[0.8.8] - 2017-05-26
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- /status endpoint showing all available models instead of only those whose name starts with *model*

[0.8.7] - 2017-05-24
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Fixed range calculation for crf #355

[0.8.6] - 2017-05-15
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Fixed duckling dimension persistence. fixes #358

[0.8.5] - 2017-05-10
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Fixed pypi installation dependencies (e.g. flask). fixes #354

[0.8.4] - 2017-05-10
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Fixed CRF model training without entities. fixes #345

[0.8.3] - 2017-05-10
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Fixed Luis emulation and added test to catch regression. Fixes #353

[0.8.2] - 2017-05-08
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- deepcopy of context #343

[0.8.1] - 2017-05-08
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- NER training reuses context inbetween requests

[0.8.0] - 2017-05-08
^^^^^^^^^^^^^^^^^^^^
Added
-----
- ngram character featurizer (allows better handling of out-of-vocab words)
- replaced pre-wired backends with more flexible pipeline definitions
- return top 10 intents with sklearn classifier `#199 <https://github.com/RasaHQ/rasa_nlu/pull/199>`_
- python type annotations for nearly all public functions
- added alternative method of defining entity synonyms
- support for arbitrary spacy language model names
- duckling components to provide normalized output for structured entities
- Conditional random field entity extraction (Markov model for entity tagging, better named entity recognition with low and medium data and similarly well at big data level)
- allow naming of trained models instead of generated model names
- dynamic check of requirements for the different components & error messages on missing dependencies
- support for using multiple entity extractors and combining results downstream

Changed
-------
- unified tokenizers, classifiers and feature extractors to implement common component interface
- ``src`` directory renamed to ``rasa_nlu``
- when loading data in a foreign format (api.ai, luis, wit) the data gets properly split into intent & entity examples
- Configuration:
    - added ``max_number_of_ngrams``
    - removed ``backend`` and added ``pipeline`` as a replacement
    - added ``luis_data_tokenizer``
    - added ``duckling_dimensions``
- parser output format changed
    from ``{"intent": "greeting", "confidence": 0.9, "entities": []}``

    to ``{"intent": {"name": "greeting", "confidence": 0.9}, "entities": []}``
- entities output format changed
    from ``{"start": 15, "end": 28, "value": "New York City", "entity": "GPE"}``

    to ``{"extractor": "ner_mitie", "processors": ["ner_synonyms"], "start": 15, "end": 28, "value": "New York City", "entity": "GPE"}``

    where ``extractor`` denotes the entity extractor that originally found an entity, and ``processor`` denotes components that alter entities, such as the synonym component.
- camel cased MITIE classes (e.g. ``MITIETokenizer`` → ``MitieTokenizer``)
- model metadata changed, see migration guide
- updated to spacy 1.7 and dropped training and loading capabilities for the spacy component (breaks existing spacy models!)
- introduced compatibility with both Python 2 and 3

Removed
-------

Fixed
-----
- properly parse ``str`` additionally to ``unicode`` `#210 <https://github.com/RasaHQ/rasa_nlu/issues/210>`_
- support entity only training `#181 <https://github.com/RasaHQ/rasa_nlu/issues/181>`_
- resolved conflicts between metadata and configuration values `#219 <https://github.com/RasaHQ/rasa_nlu/issues/219>`_
- removed tokenization when reading Luis.ai data (they changed their format) `#241 <https://github.com/RasaHQ/rasa_nlu/issues/241>`_

[0.7.4] - 2017-03-27
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed failed loading of example data after renaming attributes, i.e. "KeyError: 'entities'"

[0.7.3] - 2017-03-15
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed regression in mitie entity extraction on special characters
- fixed spacy fine tuning and entity recognition on passed language instance

[0.7.2] - 2017-03-13
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- python documentation about calling rasa NLU from python

[0.7.1] - 2017-03-10
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- mitie tokenization value generation `#207 <https://github.com/RasaHQ/rasa_nlu/pull/207>`_, thanks @cristinacaputo
- changed log file extension from ``.json`` to ``.log``, since the contained text is not proper json


[0.7.0] - 2017-03-10
^^^^^^^^^^^^^^^^^^^^
This is a major version update. Please also have a look at the `Migration Guide <https://rasa-nlu.readthedocs.io/en/latest/migrations.html>`_.

Added
-----
- Changelog ;)
- option to use multi-threading during classifier training
- entity synonym support
- proper temporary file creation during tests
- mitie_sklearn backend using mitie tokenization and sklearn classification
- option to fine-tune spacy NER models
- multithreading support of build in REST server (e.g. using gunicorn)
- multitenancy implementation to allow loading multiple models which share the same backend

Fixed
-----
- error propagation on failed vector model loading (spacy)
- escaping of special characters during mitie tokenization

[0.6-beta] - 2017-01-31
^^^^^^^^^^^^^^^^^^^^^^^

.. _`master`: https://github.com/RasaHQ/rasa_nlu/

.. _`Semantic Versioning`: http://semver.org/
