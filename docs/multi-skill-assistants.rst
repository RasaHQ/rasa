.. :desc: Iterate quickly by developing reusable building blocks of AI assistant skills
       and combining them at training time.

.. _multi_skill_bots:


Multi-skill Assistants
======================

Rasa supports building AI assistants from multiple contextual AI assistant skills.
This allows for the development of reusable building blocks of skills which you can use within
your different projects. For example, one skill could handle chitchat while another skill
is responsible for greeting your users. You can develop skills in isolation, and then
import them for the combined training of a contextual AI assistant.

An example directory structure could e.g. look like this:

.. code-block:: bash

    .
    ├── config.yml
    └── skills
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
    - skills/MoodBot

The configuration file of the ``MoodBot`` in turn references the ``GreetBot``:

.. code-block:: yaml

    imports:
    - ../GreetBot

The ``GreetBot`` skill does not specify further skills so the ``config.yml`` can be
omitted.

Rasa uses relative paths from the referencing configuration file to import skills.
These can be anywhere on your file system as long as the file access is permitted.

During the training process Rasa will import all required training files, combine
them, and train a unified AI assistant.

.. note::

    Rasa will use the policy and NLU pipeline configuration of the root project
    directory during the training. Policy or NLU configurations of imported skills will
    be ignored.

.. note::

    Equal identifiers will be merged, e.g. if two skills have training data
    for an intent ``greet``, their training data will be combined.
