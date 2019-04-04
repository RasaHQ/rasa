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
- created a common utils package ``rasa.utils`` for nlu and core, common methods like ``read_yaml`` moved there

Removed
-------

Fixed
-----
-evaluating core stories with two stage fallback gave an error, trying to handle None for a policy
