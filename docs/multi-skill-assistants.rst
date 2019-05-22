.. :desc: Iterate quickly by developing reusable building blocks of AI assistant skills
       and combining them at training time.

.. _multi-skill-assistants:

Multi-skill Assistants
======================

You can build a contextual AI assistant by combining reusable "building blocks"
called skills.
You might, for example, handle chitchat with one skill and greet your users with
another. These skills can be developed in isolation, and then combined at train time
to create your assistant.

An example directory structure could look like this:

.. code-block:: bash

    .
    ├── config.yml
    └── skills
        ├── GreetBot
        │   ├── data
        │   │   ├── nlu.md
        │   │   └── stories.md
        │   └── domain.yml
        └── ChitchatBot
            ├── config.yml
            ├── data
            │   ├── nlu.md
            │   └── stories.md
            └── domain.yml

In this example the contextual AI assistant imports the ``ChitchatBot`` skill which in turn
imports the ``GreetBot`` skill. Skill imports are defined in the configuration files of
each project. In our example, the ``config.yml`` in the root project would look like this:

.. code-block:: yaml

    imports:
    - skills/ChitchatBot

The configuration file of the ``ChitchatBot`` in turn references the ``GreetBot``:

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
