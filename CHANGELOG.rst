Change Log
==========

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 0.7.0.

[Unreleased] - `master`_
^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: This version is not yet released and is under active development.

Added
-----
- ngram character featurizer (allows better handling of out-of-vocab words)
- replaced pre-wired backends with more flexible pipeline definitions
- return top 10 intents with sklearn classifier `#199 <https://github.com/golastmile/rasa_nlu/pull/199>`_
- python type annotations for nearly all public functions

Changed
-------
- unified tokenizers, classifiers and feature extractors to implement common component interface
- ``src`` directory renamed to ``rasa_nlu``
- when loading data in a foreign format (api.ai, luis, wit) the data gets properly split into intent & entity examples
- Configuration:
    - added ``max_number_of_ngrams``
    - added ``pipeline``
    - added ``luis_data_tokenizer``
    - removed ``backend``
- parser output format changed
    from ``{"intent": "greeting", "confidence": 0.9, "entities": []}``

    to ``{"intent": {"name": "greeting", "confidence": 0.9}, "entities": []}``
- camel cased MITIE classes (e.g. ``MITIETokenizer`` → ``MitieTokenizer``)
- model metadata changed, see migration guide
- updated to spacy 1.7 (breaks existing spacy models!)
- introduced compatibility with both Python 2 and 3

Removed
-------

Fixed
-----
- properly parse ``str`` additionally to ``unicode`` `#210 <https://github.com/golastmile/rasa_nlu/issues/210>`_
- support entity only training `#181 <https://github.com/golastmile/rasa_nlu/issues/181>`_
- resolved conflicts between metadata and configuration values `#219 <https://github.com/golastmile/rasa_nlu/issues/219>`_

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
- mitie tokenization value generation `#207 <https://github.com/golastmile/rasa_nlu/pull/207>`_, thanks @cristinacaputo
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

.. _`master`: https://github.com/golastmile/rasa_nlu/

.. _`Semantic Versioning`: http://semver.org/
