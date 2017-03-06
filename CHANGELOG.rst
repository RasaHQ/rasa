Change Log
==========

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 0.7.0.

[Unreleased] - `master`_
^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: This version is not yet released and is under active development.

Added
-----
Changed
-------
Removed
-------
Fixed
-----

[0.7.0] - 2017-02-28
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
