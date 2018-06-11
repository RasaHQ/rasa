.. _debugging:

Debugging a Rasa Bot
====================


To debug your bot, run it on the command line with the ``--debug`` flag. 

For example:

.. code-block:: bash

  python -m rasa_core.run -d models/dialogue -u models/nlu/current --debug


This will print lots of information to help you understand what's going on.

