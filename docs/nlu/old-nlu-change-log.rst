:desc: Rasa NLU Changelog a
 a
.. _old-nlu-change-log: a
 a
NLU Change Log a
============== a
 a
All notable changes to this project will be documented in this file. a
This project adheres to `Semantic Versioning`_ starting with version 0.7.0. a
 a
[0.15.1] - Unreleased a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed bug in rasa_nlu.test script that appeared if no intent classifier was present a
 a
[0.15.0] - 2019-04-23 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- Added a detailed warning showing which entities are overlapping a
- Authentication token can be also set with env variable ``RASA_NLU_TOKEN``. a
- ``SpacyEntityExtractor`` supports same entity filtering as ``DucklingHTTPExtractor`` a
- **added support for python 3.7** a
 a
Changed a
------- a
- validate training data only if used for training a
- applied spacy guidelines on how to disable pipeline components a
- starter packs now also tested when attempting to merge a branch to master a
- new consistent naming scheme for pipelines: a
  - ``tensorflow_embedding`` pipeline template renamed to ``supervised_embeddings`` a
  - ``spacy_sklearn`` pipeline template renamed to ``pretrained_embeddings_spacy`` a
  - requirements files, sample configs, and dockerfiles renamed accordingly a
- ``/train`` endpoint now returns a zipfile of the trained model. a
- pipeline components in the config file should be provided a
  with their class name a
- persisted components file name changed a
- replace pep8 with pycodestyle a
- ``Component.name`` property returns component's class name a
- Components ``load(...)``, ``create(...)`` and ``cache_key(...)`` methods a
  additionally take component's meta/config dicts a
- Components ``persist(...)`` method additionally takes file name prefix a
- renamed ``rasa_nlu.evaluate`` to ``rasa_nlu.test`` a
- renamed ``rasa_nlu.test.run_cv_evaluation`` to a
  ``rasa_nlu.test.cross_validate`` a
- renamed ``rasa_nlu.train.do_train()`` to ``rasa_nlu.train.train()`` a
- train command can now also load config from file a
- updated to tensorflow 1.13 a
 a
Removed a
------- a
- **removed python 2.7 support** a
 a
Fixed a
----- a
- ``RegexFeaturizer`` detects all regex in user message (not just first) a
- do_extractors_support_overlap now correctly throws an exception only if no extractors are a
  passed or if extractors that do not support overlapping entities are used. a
- Docs entry for pretrained embeddings pipeline is now consistent with the a
  code in ``registry.py`` a
 a
 a
[0.14.6] - 2019-03-20 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed Changelog dates (dates had the wrong year attached) a
 a
[0.14.5] - 2019-03-19 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- properly tag ``stable`` docker image (instead of alpha) a
 a
[0.14.3] - 2019-02-01 a
^^^^^^^^^^^^^^^^^^^^^ a
- a
 a
Changed a
------- a
- starter packs are now tested in parallel with the unittests, a
  and only on branches ending in ``.x`` (i.e. new version releases) a
- pinned ``coloredlogs``, ``future`` and ``packaging`` a
 a
[0.14.2] - 2019-01-29 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- ``rasa_nlu.evaluate`` now exports reports into a folder and also a
  includes the entity extractor reports a
 a
Changed a
------- a
- updated requirements to match Core and SDK a
- pinned keras dependecies a
 a
[0.14.1] - 2019-01-23 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- scikit-learn is a global requirement a
 a
.. _nluv0-14-0: a
 a
[0.14.0] - 2019-01-23 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- Ability to save successful predictions and classification results to a JSON a
  file from ``rasa_nlu.evaluate`` a
- environment variables specified with ``${env_variable}`` in a yaml a
  configuration file are now replaced with the value of the environment a
  variable a
- more documentation on how to run NLU with Docker a
- ``analyzer`` parameter to ``intent_featurizer_count_vectors`` featurizer to a
  configure whether to use word or character n-grams a
- Travis script now clones and tests the Rasa NLU starter pack a
 a
Changed a
------- a
- ``EmbeddingIntentClassifier`` has been refactored, including changes to the a
  config parameters as well as comments and types for all class functions. a
- the http server's ``POST /evaluate`` endpoint returns evaluation results a
  for both entities and intents a
- replaced ``yaml`` with ``ruamel.yaml`` a
- updated spacy version to 2.0.18 a
- updated TensorFlow version to 1.12.0 a
- updated scikit-learn version to 0.20.2 a
- updated cloudpickle version to 0.6.1 a
- updated requirements to match Core and SDK a
- pinned keras dependecies a
 a
Removed a
------- a
- ``/config`` endpoint a
- removed pinning of ``msgpack`` and unused package ``python-msgpack`` a
- removed support for ``ner_duckling``. Now supports only ``ner_duckling_http`` a
 a
Fixed a
----- a
- Should loading jieba custom dictionaries only once. a
- Set attributes of custom components correctly if they defer from the default a
- NLU Server can now handle training data mit emojis in it a
- If the ``token_name`` is not given in the endpoint configuration, the default a
  value is ``token`` instead of ``None`` a
- Throws error only if ``ner_crf`` picks up overlapping entities. If the a
  entity extractor supports overlapping entitis no error is thrown. a
- Updated CORS support for the server. a
  Added the ``Access-Control-Allow-Headers`` and ``Content-Type`` headers a
  for nlu server a
- parsing of emojis which are sent within jsons a
- Bad input shape error from ``sklearn_intent_classifier`` when using a
  ``scikit-learn==0.20.2`` a
 a
[0.13.8] - 2018-11-21 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- pinned spacy version to ``spacy<=2.0.12,>2.0`` to avoid dependency conflicts a
  with tensorflow a
 a
[0.13.7] - 2018-10-11 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- ``rasa_nlu.server`` allowed more than ``max_training_processes`` a
  to be trained if they belong to different projects. a
  ``max_training_processes`` is now a global parameter, regardless of what a
  project the training process belongs to. a
 a
 a
[0.13.6] - 2018-10-04 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- ``boto3`` is now loaded lazily in ``AWSPersistor`` and is not a
  included in ``requirements_bare.txt`` anymore a
 a
Fixed a
----- a
- Allow training of pipelines containing ``EmbeddingIntentClassifier`` in a
  a separate thread on python 3. This makes http server calls to ``/train`` a
  non-blocking a
- require ``scikit-learn<0.20`` in setup py to avoid corrupted installations a
  with the most recent scikit learn a
 a
 a
[0.13.5] - 2018-09-28 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- Training data is now validated after loading from files in ``loading.py`` a
  instead of on initialisation of ``TrainingData`` object a
 a
Fixed a
----- a
- ``Project`` set up to pull models from a remote server only use a
  the pulled model instead of searching for models locally a
 a
[0.13.4] - 2018-09-19 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- pinned matplotlib to 2.x (not ready for 3.0 yet) a
- pytest-services since it wasn't used and caused issues on Windows a
 a
[0.13.3] - 2018-08-28 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- ``EndpointConfig`` class that handles authenticated requests a
  (ported from Rasa Core) a
- ``DataRouter()`` class supports a ``model_server`` ``EndpointConfig``, a
  which it regularly queries to fetch NLU models a
- this can be used with ``rasa_nlu.server`` with the ``--endpoint`` option a
  (the key for this the model server config is ``model``) a
- docs on model fetching from a URL a
- ability to specify lookup tables in training data a
 a
Changed a
------- a
- loading training data from a URL requires an instance of ``EndpointConfig`` a
 a
- Changed evaluate behaviour to plot two histogram bars per bin. a
  Plotting confidence of right predictions in a wine-ish colour a
  and wrong ones in a blue-ish colour. a
 a
Removed a
------- a
 a
Fixed a
----- a
- re-added support for entity names with special characters in markdown format a
 a
[0.13.2] - 2018-08-28 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Changed a
------- a
- added information about migrating the CRF component from 0.12 to 0.13 a
 a
Fixed a
----- a
- pipelines containing the ``EmbeddingIntentClassifier`` are not trained in a a
  separate thread, as this may lead to freezing during training a
 a
[0.13.1] - 2018-08-07 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- documentation example for creating a custom component a
 a
Fixed a
----- a
- correctly pass reference time in miliseconds to duckling_http a
 a
.. _nluv0-13-0: a
 a
[0.13.0] - 2018-08-02 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
.. warning:: a
 a
  This is a release **breaking backwards compatibility**. a
  Unfortunately, it is not possible to load previously trained models as a
  the parameters for the tensorflow and CRF models changed. a
 a
Added a
----- a
- support for `tokenizer_jieba` load custom dictionary from config a
- allow pure json including pipeline configuration on train endpoint a
- doc link to a community contribution for Rasa NLU in Chinese a
- support for component ``count_vectors_featurizer`` use ``tokens`` a
  feature provide by tokenizer a
- 2-character and a 5-character prefix features to ``ner_crf`` a
- ``ner_crf`` with whitespaced tokens to ``tensorflow_embedding`` pipeline a
- predict empty string instead of None for intent name a
- update default parameters for tensorflow embedding classifier a
- do not predict anything if feature vector contains only zeros a
  in tensorflow embedding classifier a
- change persistence keywords in tensorflow embedding classifier a
  (make previously trained models impossible to load) a
- intent_featurizer_count_vectors adds features to text_features a
  instead of overwriting them a
- add basic OOV support to intent_featurizer_count_vectors (make a
  previously trained models impossible to load) a
- add a feature for each regex in the training set for crf_entity_extractor a
- Current training processes count for server and projects. a
- the ``/version`` endpoint returns a new field ``minimum_compatible_version`` a
- added logging of intent prediction errors to evaluation script a
- added histogram of confidence scores to evaluation script a
- documentation for the ``ner_duckling_http`` component a
 a
Changed a
------- a
- renamed CRF features ``wordX`` to ``suffixX`` and ``preX`` to ``suffixX`` a
- L1 and L2 regularisation defaults in ``ner_crf`` both set to 0.1 a
- ``whitespace_tokenizer`` ignores punctuation ``.,!?`` before a
  whitespace or end of string a
- Allow multiple training processes per project a
- Changed AlreadyTrainingError to MaxTrainingError. The first one was used a
  to indicate that the project was already training. The latest will show a
  an error when the server isn't able to training more models. a
- ``Interpreter.ensure_model_compatibility`` takes a new parameters for a
  the version to compare the model version against a
- confusion matrix plot gets saved to file automatically during evaluation a
 a
Removed a
------- a
- dependence on spaCy when training ``ner_crf`` without POS features a
- documentation for the ``ner_duckling`` component - facebook doesn't maintain a
  the underlying clojure version of duckling anymore. component will be a
  removed in the next release. a
 a
Fixed a
----- a
- Fixed Luis emulation output to add start, end position and a
  confidence for each entity. a
- Fixed byte encoding issue where training data could not be a
  loaded by URL in python 3. a
 a
[0.12.3] - 2018-05-02 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- Returning used model name and project name in the response a
  of ``GET /parse`` and ``POST /parse`` as ``model`` and ``project`` a
  respectively. a
 a
Fixed a
----- a
- readded possibility to set fixed model name from http train endpoint a
 a
 a
[0.12.2] - 2018-04-20 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed duckling text extraction for ner_duckling_http a
 a
 a
[0.12.1] - 2018-04-18 a
^^^^^^^^^^^^^^^^^^^^^ a
Added a
----- a
- support for retrieving training data from a URL a
 a
Fixed a
----- a
- properly set duckling http url through environment setting a
- improvements and fixes to the configuration and pipeline a
  documentation a
 a
.. _nluv0-12-0: a
 a
[0.12.0] - 2018-04-17 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- support for inline entity synonyms in markdown training format a
- support for regex features in markdown training format a
- support for splitting and training data into multiple and mixing formats a
- support for markdown files containing regex-features or synonyms only a
- added ability to list projects in cloud storage services for model loading a
- server evaluation endpoint at ``POST /evaluate`` a
- server endpoint at ``DELETE /models`` to unload models from server memory a
- CRF entity recognizer now returns a confidence score when extracting entities a
- added count vector featurizer to create bag of words representation a
- added embedding intent classifier implemented in tensorflow a
- added tensorflow requirements a
- added docs blurb on handling contextual dialogue a
- distribute package as wheel file in addition to source a
  distribution (faster install) a
- allow a component to specify which languages it supports a
- support for persisting models to Azure Storage a
- added tokenizer for CHINESE (``zh``) as well as instructions on how to load a
  MITIE model a
 a
Changed a
------- a
- model configuration is separated from server / train configuration. This is a a
  **breaking change** and models need to be retrained. See migrations guide. a
- Regex features are now sorted internally. a
  **retrain your model if you use regex features** a
- The keyword intent classifier now returns ``null`` instead a
  of ``"None"`` as intent name in the json result if there's no match a
- in teh evaluation results, replaced ``O`` with the string a
  ``no_entity`` for better understanding a
- The ``CRFEntityExtractor`` now only trains entity examples that have a
  ``"extractor": "ner_crf"`` or no extractor at all a
- Ignore hidden files when listing projects or models a
- Docker Images now run on python 3.6 for better non-latin character set support a
- changed key name for a file in ngram featurizer a
- changed ``jsonObserver`` to generate logs without a record seperator a
- Improve jsonschema validation: text attribute of training data samples a
  can not be empty a
- made the NLU server's ``/evaluate`` endpoint asynchronous a
 a
Fixed a
----- a
- fixed certain command line arguments not getting passed into a
  the ``data_router`` a
 a
[0.11.4] - 2018-03-19 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- google analytics docs survey code a
 a
 a
[0.11.3] - 2018-02-13 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- capitalization issues during spacy named entity recognition a
 a
 a
[0.11.2] - 2018-02-06 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Formatting of tokens without assigned entities in evaluation a
 a
 a
[0.11.1] - 2018-02-02 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Changelog doc formatting a
- fixed project loading for newly added projects to a running server a
- fixed certain command line arguments not getting passed into the data_router a
 a
.. _nluv0-11-0: a
 a
[0.11.0] - 2018-01-30 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- non ascii character support for anything that gets json dumped (e.g. a
  training data received over HTTP endpoint) a
- evaluation of entity extraction performance in ``evaluation.py`` a
- support for spacy 2.0 a
- evaluation of intent classification with crossvalidation in ``evaluation.py`` a
- support for splitting training data into multiple files a
  (markdown and JSON only) a
 a
Changed a
------- a
- removed ``-e .`` from requirements files - if you want to install a
  the app use ``pip install -e .`` a
- fixed http duckling parsing for non ``en`` languages a
- fixed parsing of entities from markdown training data files a
 a
 a
[0.10.6] - 2018-01-02 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- support asterisk style annotation of examples in markdown format a
 a
Fixed a
----- a
- Preventing capitalized entities from becoming synonyms of the form a
  lower-cased → capitalized a
 a
 a
[0.10.5] - 2017-12-01 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- read token in server from config instead of data router a
- fixed reading of models with none date name prefix in server a
 a
 a
[0.10.4] - 2017-10-27 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- docker image build a
 a
 a
[0.10.3] - 2017-10-26 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- support for new dialogflow data format (previously api.ai) a
- improved support for custom components (components are a
  stored by class name in stored metadata to allow for components a
  that are not mentioned in the Rasa NLU registry) a
- language option to convert script a
 a
Fixed a
----- a
- Fixed loading of default model from S3. Fixes #633 a
- fixed permanent training status when training fails #652 a
- quick fix for None "_formatter_parser" bug a
 a
 a
[0.10.1] - 2017-10-06 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- readme issues a
- improved setup py welcome message a
 a
.. _nluv0-10-0: a
 a
[0.10.0] - 2017-09-27 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- Support for training data in Markdown format a
- Cors support. You can now specify allowed cors origins a
  within your configuration file. a
- The HTTP server is now backed by Klein (Twisted) instead of Flask. a
  The server is now asynchronous but is no more WSGI compatible a
- Improved Docker automated builds a
- Rasa NLU now works with projects instead of models. A project can a
  be the basis for a restaurant search bot in German or a customer a
  service bot in English. A model can be seen as a snapshot of a project. a
 a
Changed a
------- a
- Root project directories have been slightly rearranged to a
  clean up new docker support a
- use ``Interpreter.create(metadata, ...)`` to create interpreter a
  from dict and ``Interpreter.load(file_name, ...)`` to create a
  interpreter with metadata from a file a
- Renamed ``name`` parameter to ``project`` a
- Docs hosted on GitHub pages now: a
  `Documentation <https://rasahq.github.io/rasa_nlu>`_ a
- Adapted remote cloud storages to support projects a
  (backwards incompatible!) a
 a
Fixed a
----- a
- Fixed training data persistence. Fixes #510 a
- Fixed UTF-8 character handling when training through HTTP interface a
- Invalid handling of numbers extracted from duckling a
  during synonym handling. Fixes #517 a
- Only log a warning (instead of throwing an exception) on a
  misaligned entities during mitie NER a
 a
 a
[0.9.2] - 2017-08-16 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- removed unnecessary `ClassVar` import a
 a
 a
[0.9.1] - 2017-07-11 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- removed obsolete ``--output`` parameter of ``train.py``. a
  use ``--path`` instead. fixes #473 a
 a
.. _nluv0-9-0: a
 a
[0.9.0] - 2017-07-07 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- increased test coverage to avoid regressions (ongoing) a
- added regex featurization to support intent classification a
  and entity extraction (``intent_entity_featurizer_regex``) a
 a
Changed a
------- a
- replaced existing CRF library (python-crfsuite) with a
  sklearn-crfsuite (due to better windows support) a
- updated to spacy 1.8.2 a
- logging format of logged request now includes model name and timestamp a
- use module specific loggers instead of default python root logger a
- output format of the duckling extractor changed. the ``value`` a
  field now includes the complete value from duckling instead of a
  just text (so this is an property is an object now instead of just text). a
  includes granularity information now. a
- deprecated ``intent_examples`` and ``entity_examples`` sections in a
  training data. all examples should go into the ``common_examples`` section a
- weight training samples based on class distribution during ner_crf a
  cross validation and sklearn intent classification training a
- large refactoring of the internal training data structure and a
  pipeline architecture a
- numpy is now a required dependency a
 a
Removed a
------- a
- luis data tokenizer configuration value (not used anymore, a
  luis exports char offsets now) a
 a
Fixed a
----- a
- properly update coveralls coverage report from travis a
- persistence of duckling dimensions a
- changed default response of untrained ``intent_classifier_sklearn`` a
  from ``"intent": None`` to ``"intent": {"name": None, "confidence": 0.0}`` a
- ``/status`` endpoint showing all available models instead of only a
  those whose name starts with *model* a
- properly return training process ids #391 a
 a
 a
[0.8.12] - 2017-06-29 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed missing argument attribute error a
 a
 a
 a
[0.8.11] - 2017-06-07 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- updated mitie installation documentation a
 a
 a
[0.8.10] - 2017-05-31 a
^^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed documentation about training data format a
 a
 a
[0.8.9] - 2017-05-26 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- properly handle response_log configuration variable being set to ``null`` a
 a
 a
[0.8.8] - 2017-05-26 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- ``/status`` endpoint showing all available models instead of only a
  those whose name starts with *model* a
 a
 a
[0.8.7] - 2017-05-24 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Fixed range calculation for crf #355 a
 a
 a
[0.8.6] - 2017-05-15 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Fixed duckling dimension persistence. fixes #358 a
 a
 a
[0.8.5] - 2017-05-10 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Fixed pypi installation dependencies (e.g. flask). fixes #354 a
 a
 a
[0.8.4] - 2017-05-10 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Fixed CRF model training without entities. fixes #345 a
 a
 a
[0.8.3] - 2017-05-10 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- Fixed Luis emulation and added test to catch regression. Fixes #353 a
 a
 a
[0.8.2] - 2017-05-08 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- deepcopy of context #343 a
 a
 a
[0.8.1] - 2017-05-08 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- NER training reuses context inbetween requests a
 a
.. _nluv0-8-0: a
 a
[0.8.0] - 2017-05-08 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Added a
----- a
- ngram character featurizer (allows better handling of out-of-vocab words) a
- replaced pre-wired backends with more flexible pipeline definitions a
- return top 10 intents with sklearn classifier a
  `#199 <https://github.com/RasaHQ/rasa_nlu/pull/199>`_ a
- python type annotations for nearly all public functions a
- added alternative method of defining entity synonyms a
- support for arbitrary spacy language model names a
- duckling components to provide normalized output for structured entities a
- Conditional random field entity extraction (Markov model for entity a
  tagging, better named entity recognition with low and medium data and a
  similarly well at big data level) a
- allow naming of trained models instead of generated model names a
- dynamic check of requirements for the different components & error a
  messages on missing dependencies a
- support for using multiple entity extractors and combining results downstream a
 a
Changed a
------- a
- unified tokenizers, classifiers and feature extractors to implement a
  common component interface a
- ``src`` directory renamed to ``rasa_nlu`` a
- when loading data in a foreign format (api.ai, luis, wit) the data a
  gets properly split into intent & entity examples a
- Configuration: a
    - added ``max_number_of_ngrams`` a
    - removed ``backend`` and added ``pipeline`` as a replacement a
    - added ``luis_data_tokenizer`` a
    - added ``duckling_dimensions`` a
- parser output format changed a
    from ``{"intent": "greeting", "confidence": 0.9, "entities": []}`` a
 a
    to ``{"intent": {"name": "greeting", "confidence": 0.9}, "entities": []}`` a
- entities output format changed a
    from ``{"start": 15, "end": 28, "value": "New York City", "entity": "GPE"}`` a
 a
    to ``{"extractor": "ner_mitie", "processors": ["ner_synonyms"], "start": 15, "end": 28, "value": "New York City", "entity": "GPE"}`` a
 a
    where ``extractor`` denotes the entity extractor that originally found an entity, and ``processor`` denotes components that alter entities, such as the synonym component. a
- camel cased MITIE classes (e.g. ``MITIETokenizer`` → ``MitieTokenizer``) a
- model metadata changed, see migration guide a
- updated to spacy 1.7 and dropped training and loading capabilities for a
  the spacy component (breaks existing spacy models!) a
- introduced compatibility with both Python 2 and 3 a
 a
Fixed a
----- a
- properly parse ``str`` additionally to ``unicode`` a
  `#210 <https://github.com/RasaHQ/rasa_nlu/issues/210>`_ a
- support entity only training a
  `#181 <https://github.com/RasaHQ/rasa_nlu/issues/181>`_ a
- resolved conflicts between metadata and configuration values a
  `#219 <https://github.com/RasaHQ/rasa_nlu/issues/219>`_ a
- removed tokenization when reading Luis.ai data (they changed their format) a
  `#241 <https://github.com/RasaHQ/rasa_nlu/issues/241>`_ a
 a
 a
[0.7.4] - 2017-03-27 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed failed loading of example data after renaming attributes, a
  i.e. "KeyError: 'entities'" a
 a
 a
[0.7.3] - 2017-03-15 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- fixed regression in mitie entity extraction on special characters a
- fixed spacy fine tuning and entity recognition on passed language instance a
 a
 a
[0.7.2] - 2017-03-13 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- python documentation about calling rasa NLU from python a
 a
 a
[0.7.1] - 2017-03-10 a
^^^^^^^^^^^^^^^^^^^^ a
 a
Fixed a
----- a
- mitie tokenization value generation a
  `#207 <https://github.com/RasaHQ/rasa_nlu/pull/207>`_, thanks @cristinacaputo a
- changed log file extension from ``.json`` to ``.log``, a
  since the contained text is not proper json a
 a
.. _nluv0-7-0: a
 a
[0.7.0] - 2017-03-10 a
^^^^^^^^^^^^^^^^^^^^ a
This is a major version update. Please also have a look at the a
`Migration Guide <https://rasahq.github.io/rasa_nlu/migrations.html>`_. a
 a
Added a
----- a
- Changelog ;) a
- option to use multi-threading during classifier training a
- entity synonym support a
- proper temporary file creation during tests a
- mitie_sklearn backend using mitie tokenization and sklearn classification a
- option to fine-tune spacy NER models a
- multithreading support of build in REST server (e.g. using gunicorn) a
- multitenancy implementation to allow loading multiple models which a
  share the same backend a
 a
Fixed a
----- a
- error propagation on failed vector model loading (spacy) a
- escaping of special characters during mitie tokenization a
 a
 a
[0.6-beta] - 2017-01-31 a
^^^^^^^^^^^^^^^^^^^^^^^ a
 a
.. _`master`: https://github.com/RasaHQ/rasa_nlu/ a
 a
.. _`Semantic Versioning`: http://semver.org/ a
 a