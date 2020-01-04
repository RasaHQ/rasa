:desc: Change the way Rasa imports training data by replacing the default importer or
       writing your own importer.

.. _training-data-importers:

Training Data Importers
=======================

.. edit-link::

.. contents::
   :local:

By default, you can use command line arguments to specify where Rasa should look
for training data on your disk. Rasa then loads any potential training files and uses
them to train your assistant.

If needed, you can also customize `how` Rasa imports training data.
Potential use cases for this might be:

- using a custom parser to load training data in other formats
- using different approaches to collect training data (e.g. loading them from different resources)

You can instruct Rasa to load and use your custom importer by adding the section
``importers`` to the Rasa configuration file and specifying the importer with its
full class path:

.. code-block:: yaml

   importers:
   - name: "module.CustomImporter"
     parameter1: "value"
     parameter2: "value2"
   - name: "module.AnotherCustomImporter"

The ``name`` key is used to determine which importer should be loaded. Any extra
parameters are passed as constructor arguments to the loaded importer.

.. note::

   You can specify multiple importers. Rasa will automatically merge their results.


RasaFileImporter (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default Rasa uses the importer ``RasaFileImporter``. If you want to use it on its
own, you don't have to specify anything in your configuration file.
If you want to use it together with other importers, add it to your
configuration file:

.. code-block:: yaml

   importers:
   - name: "RasaFileImporter"

MultiProjectImporter (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    This feature is currently experimental and might change or be removed in the future.
    Please share your feedback on it in the `forum <https://forum.rasa.com>`_ to help
    us making this feature ready for production.

With this importer you can build a contextual AI assistant by combining multiple
reusable Rasa projects.
You might, for example, handle chitchat with one project and greet your users with
another. These projects can be developed in isolation, and then combined at train time
to create your assistant.

An example directory structure could look like this:

.. code-block:: bash

    .
    ├── config.yml
    └── projects
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

In this example the contextual AI assistant imports the ``ChitchatBot`` project which in turn
imports the ``GreetBot`` project. Project imports are defined in the configuration files of
each project.
To instruct Rasa to use the ``MultiProjectImporter`` module, put this section in the config
file of your root project:

.. code-block:: yaml

    importers:
    - name: MultiProjectImporter


Then specify which projects you want to import.
In our example, the ``config.yml`` in the root project would look like this:

.. code-block:: yaml

    imports:
    - projects/ChitchatBot

The configuration file of the ``ChitchatBot`` in turn references the ``GreetBot``:

.. code-block:: yaml

    imports:
    - ../GreetBot

The ``GreetBot`` project does not specify further projects so the ``config.yml`` can be
omitted.

Rasa uses relative paths from the referencing configuration file to import projects.
These can be anywhere on your file system as long as the file access is permitted.

During the training process Rasa will import all required training files, combine
them, and train a unified AI assistant. The merging of the training data happens during
runtime, so no additional files with training data are created or visible.

.. note::

    Rasa will use the policy and NLU pipeline configuration of the root project
    directory during training. **Policy or NLU configurations of imported projects
    will be ignored.**

.. note::

    Equal intents, entities, slots, templates, actions and forms will be merged,
    e.g. if two projects have training data for an intent ``greet``,
    their training data will be combined.

Writing a Custom Importer
~~~~~~~~~~~~~~~~~~~~~~~~~
If you are writing a custom importer, this importer has to implement the interface of
:ref:`training-data-importers-trainingFileImporter`:

.. code-block:: python

    from typing import Optional, Text, Dict, List, Union

    import rasa
    from rasa.core.domain import Domain
    from rasa.core.interpreter import RegexInterpreter, NaturalLanguageInterpreter
    from rasa.core.training.structures import StoryGraph
    from rasa.importers.importer import TrainingDataImporter
    from rasa.nlu.training_data import TrainingData


    class MyImporter(TrainingDataImporter):
        """Example implementation of a custom importer component."""

        def __init__(
            self,
            config_file: Optional[Text] = None,
            domain_path: Optional[Text] = None,
            training_data_paths: Optional[Union[List[Text], Text]] = None,
            **kwargs: Dict
        ):
            """Constructor of your custom file importer.

            Args:
                config_file: Path to configuration file from command line arguments.
                domain_path: Path to domain file from command line arguments.
                training_data_paths: Path to training files from command line arguments.
                **kwargs: Extra parameters passed through configuration in configuration file.
            """

            pass

        async def get_domain(self) -> Domain:
            path_to_domain_file = self._custom_get_domain_file()
            return Domain.load(path_to_domain_file)

        def _custom_get_domain_file(self) -> Text:
            pass

        async def get_stories(
            self,
            interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
            template_variables: Optional[Dict] = None,
            use_e2e: bool = False,
            exclusion_percentage: Optional[int] = None,
        ) -> StoryGraph:
            from rasa.core.training.dsl import StoryFileReader

            path_to_stories = self._custom_get_story_file()
            return await StoryFileReader.read_from_file(path_to_stories, await self.get_domain())

        def _custom_get_story_file(self) -> Text:
            pass

        async def get_config(self) -> Dict:
            path_to_config = self._custom_get_config_file()
            return rasa.utils.io.read_config_file(path_to_config)

        def _custom_get_config_file(self) -> Text:
            pass

        async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
            from rasa.nlu.training_data import loading

            path_to_nlu_file = self._custom_get_nlu_file()
            return loading.load_data(path_to_nlu_file)

        def _custom_get_nlu_file(self) -> Text:
            pass



.. _training-data-importers-trainingFileImporter:

TrainingDataImporter
~~~~~~~~~~~~~~~~~~~~


.. autoclass:: rasa.importers.importer.TrainingDataImporter

   .. automethod:: get_domain

   .. automethod:: get_config

   .. automethod:: get_nlu_data

   .. automethod:: get_stories
