:desc: Iterate quickly by developing reusable building blocks of AI assistant skills
       and combining them at training time.

.. _multi_skill_bots:


Multi Skill Assistants
======================

Rasa supports building AI assistants from multiple contextual AI assistant projects.
This allows for the development of reusable building blocks of skills which you can use within
your different projects. For example, one skill could handle chitchat while another skill
is responsible for greeting your users. You can develop each skill isolated and then
import them for the combined training of a contextual AI assistant.

An example directory structure could e.g. look like this:

.. code-block:: bash

    .
    ├── config.yml
    └── data
        ├── GreetBot
        │   ├── data
        │   │   ├── nlu.md
        │   │   └── stories.md
        │   └── domain.yml
        └── MoodBot
            ├── config.yml
            ├── data
            │   ├── nlu.md
            │   └── stories.md
            └── domain.yml

In this example the contextual AI assistant imports the ``MoodBot`` skill which in turn
imports the ``GreetBot`` skill. Skill imports are defined in the configuration files of
each project. The ``config.yml`` in the root project e.g. looks like this:

.. code-block:: yaml

    imports:
    - data/MoodBot

The configuration file of the ``MoodBot`` in turn references the ``GreetBot``:

.. code-block:: yaml

    imports:
    - ../GreetBot

Rasa uses relative paths from the referencing configuration files to import skills.
These can be anywhere on your file system as long as the file access is permitted.

During the training process Rasa will import all required training files, combine
them, and train a unified AI assistant.

.. note::

    Note that equal identifiers will be merged, e.g. if two skills have training data
    for an intent ``greet``, their training data will be combined.

.. include:: feedback.inc
