:desc: Rasa NLU Changelog

.. _old-nlu-change-log:

NLU Change Log
==============

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 0.7.0.

[0.15.1] - Unreleased
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed bug in rasa_nlu.test script that appeared if no intent classifier was present

[0.15.0] - 2019-04-23
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- Added a detailed warning showing which entities are overlapping
- Authentication token can be also set with env variable ``RASA_NLU_TOKEN``.
- ``SpacyEntityExtractor`` supports same entity filtering as ``DucklingHTTPExtractor``
- **added support for python 3.7**

Changed
-------
- validate training data only if used for training
- applied spacy guidelines on how to disable pipeline components
- starter packs now also tested when attempting to merge a branch to master
- new consistent naming scheme for pipelines:
  - ``tensorflow_embedding`` pipeline template renamed to ``supervised_embeddings``
  - ``spacy_sklearn`` pipeline template renamed to ``pretrained_embeddings_spacy``
  - requirements files, sample configs, and dockerfiles renamed accordingly
- ``/train`` endpoint now returns a zipfile of the trained model.
- pipeline components in the config file should be provided
  with their class name
- persisted components file name changed
- replace pep8 with pycodestyle
- ``Component.name`` property returns component's class name
- Components ``load(...)``, ``create(...)`` and ``cache_key(...)`` methods
  additionally take component's meta/config dicts
- Components ``persist(...)`` method additionally takes file name prefix
- renamed ``rasa_nlu.evaluate`` to ``rasa_nlu.test``
- renamed ``rasa_nlu.test.run_cv_evaluation`` to
  ``rasa_nlu.test.cross_validate``
- renamed ``rasa_nlu.train.do_train()`` to ``rasa_nlu.train.train()``
- train command can now also load config from file
- updated to tensorflow 1.13

Removed
-------
- **removed python 2.7 support**

Fixed
-----
- ``RegexFeaturizer`` detects all regex in user message (not just first)
- do_extractors_support_overlap now correctly throws an exception only if no extractors are
  passed or if extractors that do not support overlapping entities are used.
- Docs entry for pretrained embeddings pipeline is now consistent with the
  code in ``registry.py``


[0.14.6] - 2019-03-20
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed Changelog dates (dates had the wrong year attached)

[0.14.5] - 2019-03-19
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- properly tag ``stable`` docker image (instead of alpha)

[0.14.3] - 2019-02-01
^^^^^^^^^^^^^^^^^^^^^
-

Changed
-------
- starter packs are now tested in parallel with the unittests,
  and only on branches ending in ``.x`` (i.e. new version releases)
- pinned ``coloredlogs``, ``future`` and ``packaging``

[0.14.2] - 2019-01-29
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``rasa_nlu.evaluate`` now exports reports into a folder and also
  includes the entity extractor reports

Changed
-------
- updated requirements to match Core and SDK
- pinned keras dependecies

[0.14.1] - 2019-01-23
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- scikit-learn is a global requirement

.. _nluv0-14-0:

[0.14.0] - 2019-01-23
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- Ability to save successful predictions and classification results to a JSON
  file from ``rasa_nlu.evaluate``
- environment variables specified with ``${env_variable}`` in a yaml
  configuration file are now replaced with the value of the environment
  variable
- more documentation on how to run NLU with Docker
- ``analyzer`` parameter to ``intent_featurizer_count_vectors`` featurizer to
  configure whether to use word or character n-grams
- Travis script now clones and tests the Rasa NLU starter pack

Changed
-------
- ``EmbeddingIntentClassifier`` has been refactored, including changes to the
  config parameters as well as comments and types for all class functions.
- the http server's ``POST /evaluate`` endpoint returns evaluation results
  for both entities and intents
- replaced ``yaml`` with ``ruamel.yaml``
- updated spacy version to 2.0.18
- updated TensorFlow version to 1.12.0
- updated scikit-learn version to 0.20.2
- updated cloudpickle version to 0.6.1
- updated requirements to match Core and SDK
- pinned keras dependecies

Removed
-------
- ``/config`` endpoint
- removed pinning of ``msgpack`` and unused package ``python-msgpack``
- removed support for ``ner_duckling``. Now supports only ``ner_duckling_http``

Fixed
-----
- Should loading jieba custom dictionaries only once.
- Set attributes of custom components correctly if they defer from the default
- NLU Server can now handle training data mit emojis in it
- If the ``token_name`` is not given in the endpoint configuration, the default
  value is ``token`` instead of ``None``
- Throws error only if ``ner_crf`` picks up overlapping entities. If the
  entity extractor supports overlapping entitis no error is thrown.
- Updated CORS support for the server.
  Added the ``Access-Control-Allow-Headers`` and ``Content-Type`` headers
  for nlu server
- parsing of emojis which are sent within jsons
- Bad input shape error from ``sklearn_intent_classifier`` when using
  ``scikit-learn==0.20.2``

[0.13.8] - 2018-11-21
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- pinned spacy version to ``spacy<=2.0.12,>2.0`` to avoid dependency conflicts
  with tensorflow

[0.13.7] - 2018-10-11
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- ``rasa_nlu.server`` allowed more than ``max_training_processes``
  to be trained if they belong to different projects.
  ``max_training_processes`` is now a global parameter, regardless of what
  project the training process belongs to.


[0.13.6] - 2018-10-04
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- ``boto3`` is now loaded lazily in ``AWSPersistor`` and is not
  included in ``requirements_bare.txt`` anymore

Fixed
-----
- Allow training of pipelines containing ``EmbeddingIntentClassifier`` in
  a separate thread on python 3. This makes http server calls to ``/train``
  non-blocking
- require ``scikit-learn<0.20`` in setup py to avoid corrupted installations
  with the most recent scikit learn


[0.13.5] - 2018-09-28
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- Training data is now validated after loading from files in ``loading.py``
  instead of on initialisation of ``TrainingData`` object

Fixed
-----
- ``Project`` set up to pull models from a remote server only use
  the pulled model instead of searching for models locally

[0.13.4] - 2018-09-19
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- pinned matplotlib to 2.x (not ready for 3.0 yet)
- pytest-services since it wasn't used and caused issues on Windows

[0.13.3] - 2018-08-28
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``EndpointConfig`` class that handles authenticated requests
  (ported from Rasa Core)
- ``DataRouter()`` class supports a ``model_server`` ``EndpointConfig``,
  which it regularly queries to fetch NLU models
- this can be used with ``rasa_nlu.server`` with the ``--endpoint`` option
  (the key for this the model server config is ``model``)
- docs on model fetching from a URL
- ability to specify lookup tables in training data

Changed
-------
- loading training data from a URL requires an instance of ``EndpointConfig``

- Changed evaluate behaviour to plot two histogram bars per bin.
  Plotting confidence of right predictions in a wine-ish colour
  and wrong ones in a blue-ish colour.

Removed
-------

Fixed
-----
- re-added support for entity names with special characters in markdown format

[0.13.2] - 2018-08-28
^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- added information about migrating the CRF component from 0.12 to 0.13

Fixed
-----
- pipelines containing the ``EmbeddingIntentClassifier`` are not trained in a
  separate thread, as this may lead to freezing during training

[0.13.1] - 2018-08-07
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- documentation example for creating a custom component

Fixed
-----
- correctly pass reference time in miliseconds to duckling_http

.. _nluv0-13-0:

[0.13.0] - 2018-08-02
^^^^^^^^^^^^^^^^^^^^^

.. warning::

  This is a release **breaking backwards compatibility**.
  Unfortunately, it is not possible to load previously trained models as
  the parameters for the tensorflow and CRF models changed.

Added
-----
- support for `tokenizer_jieba` load custom dictionary from config
- allow pure json including pipeline configuration on train endpoint
- doc link to a community contribution for Rasa NLU in Chinese
- support for component ``count_vectors_featurizer`` use ``tokens``
  feature provide by tokenizer
- 2-character and a 5-character prefix features to ``ner_crf``
- ``ner_crf`` with whitespaced tokens to ``tensorflow_embedding`` pipeline
- predict empty string instead of None for intent name
- update default parameters for tensorflow embedding classifier
- do not predict anything if feature vector contains only zeros
  in tensorflow embedding classifier
- change persistence keywords in tensorflow embedding classifier
  (make previously trained models impossible to load)
- intent_featurizer_count_vectors adds features to text_features
  instead of overwriting them
- add basic OOV support to intent_featurizer_count_vectors (make
  previously trained models impossible to load)
- add a feature for each regex in the training set for crf_entity_extractor
- Current training processes count for server and projects.
- the ``/version`` endpoint returns a new field ``minimum_compatible_version``
- added logging of intent prediction errors to evaluation script
- added histogram of confidence scores to evaluation script
- documentation for the ``ner_duckling_http`` component

Changed
-------
- renamed CRF features ``wordX`` to ``suffixX`` and ``preX`` to ``suffixX``
- L1 and L2 regularisation defaults in ``ner_crf`` both set to 0.1
- ``whitespace_tokenizer`` ignores punctuation ``.,!?`` before
  whitespace or end of string
- Allow multiple training processes per project
- Changed AlreadyTrainingError to MaxTrainingError. The first one was used
  to indicate that the project was already training. The latest will show
  an error when the server isn't able to training more models.
- ``Interpreter.ensure_model_compatibility`` takes a new parameters for
  the version to compare the model version against
- confusion matrix plot gets saved to file automatically during evaluation

Removed
-------
- dependence on spaCy when training ``ner_crf`` without POS features
- documentation for the ``ner_duckling`` component - facebook doesn't maintain
  the underlying clojure version of duckling anymore. component will be
  removed in the next release.

Fixed
-----
- Fixed Luis emulation output to add start, end position and
  confidence for each entity.
- Fixed byte encoding issue where training data could not be
  loaded by URL in python 3.

[0.12.3] - 2018-05-02
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- Returning used model name and project name in the response
  of ``GET /parse`` and ``POST /parse`` as ``model`` and ``project``
  respectively.

Fixed
-----
- readded possibility to set fixed model name from http train endpoint


[0.12.2] - 2018-04-20
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed duckling text extraction for ner_duckling_http


[0.12.1] - 2018-04-18
^^^^^^^^^^^^^^^^^^^^^
Added
-----
- support for retrieving training data from a URL

Fixed
-----
- properly set duckling http url through environment setting
- improvements and fixes to the configuration and pipeline
  documentation

.. _nluv0-12-0:

[0.12.0] - 2018-04-17
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- support for inline entity synonyms in markdown training format
- support for regex features in markdown training format
- support for splitting and training data into multiple and mixing formats
- support for markdown files containing regex-features or synonyms only
- added ability to list projects in cloud storage services for model loading
- server evaluation endpoint at ``POST /evaluate``
- server endpoint at ``DELETE /models`` to unload models from server memory
- CRF entity recognizer now returns a confidence score when extracting entities
- added count vector featurizer to create bag of words representation
- added embedding intent classifier implemented in tensorflow
- added tensorflow requirements
- added docs blurb on handling contextual dialogue
- distribute package as wheel file in addition to source
  distribution (faster install)
- allow a component to specify which languages it supports
- support for persisting models to Azure Storage
- added tokenizer for CHINESE (``zh``) as well as instructions on how to load
  MITIE model

Changed
-------
- model configuration is separated from server / train configuration. This is a
  **breaking change** and models need to be retrained. See migrations guide.
- Regex features are now sorted internally.
  **retrain your model if you use regex features**
- The keyword intent classifier now returns ``null`` instead
  of ``"None"`` as intent name in the json result if there's no match
- in teh evaluation results, replaced ``O`` with the string
  ``no_entity`` for better understanding
- The ``CRFEntityExtractor`` now only trains entity examples that have
  ``"extractor": "ner_crf"`` or no extractor at all
- Ignore hidden files when listing projects or models
- Docker Images now run on python 3.6 for better non-latin character set support
- changed key name for a file in ngram featurizer
- changed ``jsonObserver`` to generate logs without a record seperator
- Improve jsonschema validation: text attribute of training data samples
  can not be empty
- made the NLU server's ``/evaluate`` endpoint asynchronous

Fixed
-----
- fixed certain command line arguments not getting passed into
  the ``data_router``

[0.11.4] - 2018-03-19
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- google analytics docs survey code


[0.11.3] - 2018-02-13
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- capitalization issues during spacy named entity recognition


[0.11.2] - 2018-02-06
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Formatting of tokens without assigned entities in evaluation


[0.11.1] - 2018-02-02
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- Changelog doc formatting
- fixed project loading for newly added projects to a running server
- fixed certain command line arguments not getting passed into the data_router

.. _nluv0-11-0:

[0.11.0] - 2018-01-30
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- non ascii character support for anything that gets json dumped (e.g.
  training data received over HTTP endpoint)
- evaluation of entity extraction performance in ``evaluation.py``
- support for spacy 2.0
- evaluation of intent classification with crossvalidation in ``evaluation.py``
- support for splitting training data into multiple files
  (markdown and JSON only)

Changed
-------
- removed ``-e .`` from requirements files - if you want to install
  the app use ``pip install -e .``
- fixed http duckling parsing for non ``en`` languages
- fixed parsing of entities from markdown training data files


[0.10.6] - 2018-01-02
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- support asterisk style annotation of examples in markdown format

Fixed
-----
- Preventing capitalized entities from becoming synonyms of the form
  lower-cased → capitalized


[0.10.5] - 2017-12-01
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- read token in server from config instead of data router
- fixed reading of models with none date name prefix in server


[0.10.4] - 2017-10-27
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- docker image build


[0.10.3] - 2017-10-26
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- support for new dialogflow data format (previously api.ai)
- improved support for custom components (components are
  stored by class name in stored metadata to allow for components
  that are not mentioned in the Rasa NLU registry)
- language option to convert script

Fixed
-----
- Fixed loading of default model from S3. Fixes #633
- fixed permanent training status when training fails #652
- quick fix for None "_formatter_parser" bug


[0.10.1] - 2017-10-06
^^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- readme issues
- improved setup py welcome message

.. _nluv0-10-0:

[0.10.0] - 2017-09-27
^^^^^^^^^^^^^^^^^^^^^

Added
-----
- Support for training data in Markdown format
- Cors support. You can now specify allowed cors origins
  within your configuration file.
- The HTTP server is now backed by Klein (Twisted) instead of Flask.
  The server is now asynchronous but is no more WSGI compatible
- Improved Docker automated builds
- Rasa NLU now works with projects instead of models. A project can
  be the basis for a restaurant search bot in German or a customer
  service bot in English. A model can be seen as a snapshot of a project.

Changed
-------
- Root project directories have been slightly rearranged to
  clean up new docker support
- use ``Interpreter.create(metadata, ...)`` to create interpreter
  from dict and ``Interpreter.load(file_name, ...)`` to create
  interpreter with metadata from a file
- Renamed ``name`` parameter to ``project``
- Docs hosted on GitHub pages now:
  `Documentation <https://rasahq.github.io/rasa_nlu>`_
- Adapted remote cloud storages to support projects
  (backwards incompatible!)

Fixed
-----
- Fixed training data persistence. Fixes #510
- Fixed UTF-8 character handling when training through HTTP interface
- Invalid handling of numbers extracted from duckling
  during synonym handling. Fixes #517
- Only log a warning (instead of throwing an exception) on
  misaligned entities during mitie NER


[0.9.2] - 2017-08-16
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- removed unnecessary `ClassVar` import


[0.9.1] - 2017-07-11
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- removed obsolete ``--output`` parameter of ``train.py``.
  use ``--path`` instead. fixes #473

.. _nluv0-9-0:

[0.9.0] - 2017-07-07
^^^^^^^^^^^^^^^^^^^^

Added
-----
- increased test coverage to avoid regressions (ongoing)
- added regex featurization to support intent classification
  and entity extraction (``intent_entity_featurizer_regex``)

Changed
-------
- replaced existing CRF library (python-crfsuite) with
  sklearn-crfsuite (due to better windows support)
- updated to spacy 1.8.2
- logging format of logged request now includes model name and timestamp
- use module specific loggers instead of default python root logger
- output format of the duckling extractor changed. the ``value``
  field now includes the complete value from duckling instead of
  just text (so this is an property is an object now instead of just text).
  includes granularity information now.
- deprecated ``intent_examples`` and ``entity_examples`` sections in
  training data. all examples should go into the ``common_examples`` section
- weight training samples based on class distribution during ner_crf
  cross validation and sklearn intent classification training
- large refactoring of the internal training data structure and
  pipeline architecture
- numpy is now a required dependency

Removed
-------
- luis data tokenizer configuration value (not used anymore,
  luis exports char offsets now)

Fixed
-----
- properly update coveralls coverage report from travis
- persistence of duckling dimensions
- changed default response of untrained ``intent_classifier_sklearn``
  from ``"intent": None`` to ``"intent": {"name": None, "confidence": 0.0}``
- ``/status`` endpoint showing all available models instead of only
  those whose name starts with *model*
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
-----
- properly handle response_log configuration variable being set to ``null``


[0.8.8] - 2017-05-26
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- ``/status`` endpoint showing all available models instead of only
  those whose name starts with *model*


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

.. _nluv0-8-0:

[0.8.0] - 2017-05-08
^^^^^^^^^^^^^^^^^^^^

Added
-----
- ngram character featurizer (allows better handling of out-of-vocab words)
- replaced pre-wired backends with more flexible pipeline definitions
- return top 10 intents with sklearn classifier
  `#199 <https://github.com/RasaHQ/rasa_nlu/pull/199>`_
- python type annotations for nearly all public functions
- added alternative method of defining entity synonyms
- support for arbitrary spacy language model names
- duckling components to provide normalized output for structured entities
- Conditional random field entity extraction (Markov model for entity
  tagging, better named entity recognition with low and medium data and
  similarly well at big data level)
- allow naming of trained models instead of generated model names
- dynamic check of requirements for the different components & error
  messages on missing dependencies
- support for using multiple entity extractors and combining results downstream

Changed
-------
- unified tokenizers, classifiers and feature extractors to implement
  common component interface
- ``src`` directory renamed to ``rasa_nlu``
- when loading data in a foreign format (api.ai, luis, wit) the data
  gets properly split into intent & entity examples
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
- updated to spacy 1.7 and dropped training and loading capabilities for
  the spacy component (breaks existing spacy models!)
- introduced compatibility with both Python 2 and 3

Fixed
-----
- properly parse ``str`` additionally to ``unicode``
  `#210 <https://github.com/RasaHQ/rasa_nlu/issues/210>`_
- support entity only training
  `#181 <https://github.com/RasaHQ/rasa_nlu/issues/181>`_
- resolved conflicts between metadata and configuration values
  `#219 <https://github.com/RasaHQ/rasa_nlu/issues/219>`_
- removed tokenization when reading Luis.ai data (they changed their format)
  `#241 <https://github.com/RasaHQ/rasa_nlu/issues/241>`_


[0.7.4] - 2017-03-27
^^^^^^^^^^^^^^^^^^^^

Fixed
-----
- fixed failed loading of example data after renaming attributes,
  i.e. "KeyError: 'entities'"


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
- mitie tokenization value generation
  `#207 <https://github.com/RasaHQ/rasa_nlu/pull/207>`_, thanks @cristinacaputo
- changed log file extension from ``.json`` to ``.log``,
  since the contained text is not proper json

.. _nluv0-7-0:

[0.7.0] - 2017-03-10
^^^^^^^^^^^^^^^^^^^^
This is a major version update. Please also have a look at the
`Migration Guide <https://rasahq.github.io/rasa_nlu/migrations.html>`_.

Added
-----
- Changelog ;)
- option to use multi-threading during classifier training
- entity synonym support
- proper temporary file creation during tests
- mitie_sklearn backend using mitie tokenization and sklearn classification
- option to fine-tune spacy NER models
- multithreading support of build in REST server (e.g. using gunicorn)
- multitenancy implementation to allow loading multiple models which
  share the same backend

Fixed
-----
- error propagation on failed vector model loading (spacy)
- escaping of special characters during mitie tokenization


[0.6-beta] - 2017-01-31
^^^^^^^^^^^^^^^^^^^^^^^

.. _`master`: https://github.com/RasaHQ/rasa_nlu/

.. _`Semantic Versioning`: http://semver.org/
