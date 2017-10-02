Change Log
==========

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning`_ starting with version 0.2.0.

[Unreleased] - `master`_
^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: This version is not yet released and is under active development.


Added
-----
- support for Rasa Core usage as a server with remote action execution

Changed
-------
- switched to max code line length 80
- removed action id - use ``action.name()`` instead. if an action implementation overrides the name, it should include the ``action_`` prefix (as it is not automatically added anymore)
- renamed ``rasa_dm.util`` to ``rasa_dm.utils``
- renamed the whole package to ``rasa_core`` (so ``rasa_dm`` is gone!)
- renamed ``Reminder`` attribute ``id`` to ``name``

Removed
-------

Fixed
-----

[0.7.0a3] - 2017-09-20
^^^^^^^^^^^^^^^^^^^^^^

Changed
-------
- use hashing when writing memorized turns into persistence - requires retraining of all models that are trained with a version prior to this
- changed ``agent.handle_message(...)`` interface for easier usage

[0.6.0a1] - 2017-08-27
^^^^^^^^^^^^^^^^^^^^^^

Added
-----
- support for multiple policies (e.g. one memoization and a Keras policy at the same time)
- loading domains from yaml files instead of defining them with python code
- added an api layer (called ``Agent``) for you to use for 95% of the things you want to do (training, persistence, loading models)
- support for reminders

Changed
-------
- large refactoring of code base

[0.5.0a1] - 2017-06-18
^^^^^^^^^^^^^^^^^^^^^^

Added
-----
- ``ScoringPolicy`` added to policy implementations (less strict than standard default policy)
- ``RasaNLUInterpreter`` to run a nlu instance within dm (instead of using the http interface)
- more tests

Changed
-------
- ``UserUtterance`` now holds the complete parse data from nlu (e.g. to access attributes other than entities or intent)
- ``Turn`` has a reference to a ``UserUtterance`` instead of directly storing intent & entities (allows access to other data)
- Simplified interface of output channels
- order of actions in the DefaultPolicy in ``possible_actions`` (``ActionListen`` now always has index 0)

Fixed
-----
- ``RedisTrackerStore`` checks if tracker is stored before accessing it (otherwise a ``None`` access exception is thrown)
- ``RegexInterpreter`` checks if the regex actually matches the message instead of assuming it always does
- ``str`` implementation for all events
- ``Controller`` can be started without an input channel (e.g. messages need to be fed into the queue manually)

[0.2.0a1] - 2017-05-18
^^^^^^^^^^^^^^^^^^^^^^
First released version.


.. _`master`: https://github.com/RasaHQ/rasa_core/

.. _`Semantic Versioning`: http://semver.org/
