:desc: Rasa Core Changelog

.. include:: ../CHANGELOG.rst
Changed
-------
- Change payloads from:
{
  "sender": <>,
  "text": <>
}

to:

{
  "sender": <>,
  "message": <>
}

in files: server.yml, docs/connectors.rst, rasa_core/server.py, rasa_core/training/interactive.py, tests/test_interactive.py