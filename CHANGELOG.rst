:desc: Rasa Changelog

Rasa Change Log
===============

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 1.0.

[Unreleased 0.15.0.aX] - `master`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Added
-----

Changed
-------
- changed removing punctuation logic in ``WhitespaceTokenizer``
- ``training_processes`` in the Rasa NLU data router have been renamed to ``pool_processes``

Removed
-------

Fixed
-----
- the ``/evaluate`` route for the Rasa NLU server now runs evaluation
  in a parallel process, which prevents the currently loaded model unloading
