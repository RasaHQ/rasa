:desc: Rasa Core Changelog

.. include:: ../CHANGELOG.rst
Added
-------
- Added `priority` property of policies to influence best policy in the case of equal confidence

Changed
-------
- Change payloads from "text" to "message" in files: server.yml, docs/connectors.rst, rasa_core/server.py, rasa_core/training/interactive.py, tests/test_interactive.py
