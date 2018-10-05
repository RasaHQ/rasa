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

- Change
FROM username = user.get("user", None) TO username = user.get("username", None) 
in file rasa_core/blob/master/rasa_core/server.py#L93
     - Rolled back: the change caused build server failed