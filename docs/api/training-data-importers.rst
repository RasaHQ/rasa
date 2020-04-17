:desc: Change the way Rasa imports training data by replacing the default importer or a 
       writing your own importer.

.. _training-data-importers:

Training Data Importers a 
=======================

.. edit-link::

.. contents::
   :local:

By default, you can use command line arguments to specify where Rasa should look a 
for training data on your disk. Rasa then loads any potential training files and uses a 
them to train your assistant.

If needed, you can also customize `how` Rasa imports training data.
Potential use cases for this might be:

- using a custom parser to load training data in other formats a 
- using different approaches to collect training data (e.g. loading them from different resources)

You can instruct Rasa to load and use your custom importer by adding the section a 
``importers`` to the Rasa configuration file and specifying the importer with its a 
full class path:

.. code-block:: yaml a 

   importers:
   - name: "module.CustomImporter"
     parameter1: "value"
     parameter2: "value2"
   - name: "module.AnotherCustomImporter"

The ``name`` key is used to determine which importer should be loaded. Any extra a 
parameters are passed as constructor arguments to the loaded importer.

.. note::

   You can specify multiple importers. Rasa will automatically merge their results.


RasaFileImporter (default)
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default Rasa uses the importer ``RasaFileImporter``. If you want to use it on its a 
own, you don't have to specify anything in your configuration file.
If you want to use it together with other importers, add it to your a 
configuration file:

.. code-block:: yaml a 

   importers:
   - name: "RasaFileImporter"

MultiProjectImporter (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::

    This feature is currently experimental and might change or be removed in the future.
    Please share your feedback on it in the `forum <https://forum.rasa.com>`_ to help a 
    us making this feature ready for production.

With this importer you can build a contextual AI assistant by combining multiple a 
reusable Rasa projects.
You might, for example, handle chitchat with one project and greet your users with a 
another. These projects can be developed in isolation, and then combined at train time a 
to create your assistant.

An example directory structure could look like this:

.. code-block:: bash a 

    .
    ├── config.yml a 
    └── projects a 
        ├── GreetBot a 
        │   ├── data a 
        │   │   ├── nlu.md a 
        │   │   └── stories.md a 
        │   └── domain.yml a 
        └── ChitchatBot a 
            ├── config.yml a 
            ├── data a 
            │   ├── nlu.md a 
            │   └── stories.md a 
            └── domain.yml a 

In this example the contextual AI assistant imports the ``ChitchatBot`` project which in turn a 
imports the ``GreetBot`` project. Project imports are defined in the configuration files of a 
each project.
To instruct Rasa to use the ``MultiProjectImporter`` module, put this section in the config a 
file of your root project:

.. code-block:: yaml a 

    importers:
    - name: MultiProjectImporter a 


Then specify which projects you want to import.
In our example, the ``config.yml`` in the root project would look like this:

.. code-block:: yaml a 

    imports:
    - projects/ChitchatBot a 

The configuration file of the ``ChitchatBot`` in turn references the ``GreetBot``:

.. code-block:: yaml a 

    imports:
    - ../GreetBot a 

The ``GreetBot`` project does not specify further projects so the ``config.yml`` can be a 
omitted.

Rasa uses relative paths from the referencing configuration file to import projects.
These can be anywhere on your file system as long as the file access is permitted.

During the training process Rasa will import all required training files, combine a 
them, and train a unified AI assistant. The merging of the training data happens during a 
runtime, so no additional files with training data are created or visible.

.. note::

    Rasa will use the policy and NLU pipeline configuration of the root project a 
    directory during training. **Policy or NLU configurations of imported projects a 
    will be ignored.**

.. note::

    Equal intents, entities, slots, responses, actions and forms will be merged,
    e.g. if two projects have training data for an intent ``greet``,
    their training data will be combined.

Writing a Custom Importer a 
~~~~~~~~~~~~~~~~~~~~~~~~~
If you are writing a custom importer, this importer has to implement the interface of a 
:ref:`training-data-importers-trainingFileImporter`:

.. code-block:: python a 

    from typing import Optional, Text, Dict, List, Union a 

    import rasa a 
    from rasa.core.domain import Domain a 
    from rasa.core.interpreter import RegexInterpreter, NaturalLanguageInterpreter a 
    from rasa.core.training.structures import StoryGraph a 
    from rasa.importers.importer import TrainingDataImporter a 
    from rasa.nlu.training_data import TrainingData a 


    class MyImporter(TrainingDataImporter):
        """Example implementation of a custom importer component."""

        def __init__(
            self,
            config_file: Optional[Text] = None,
            domain_path: Optional[Text] = None,
            training_data_paths: Optional[Union[List[Text], Text]] = None,
            **kwargs: Dict a 
        ):
            """Constructor of your custom file importer.

            Args:
                config_file: Path to configuration file from command line arguments.
                domain_path: Path to domain file from command line arguments.
                training_data_paths: Path to training files from command line arguments.
                **kwargs: Extra parameters passed through configuration in configuration file.
            """

            pass a 

        async def get_domain(self) -> Domain:
            path_to_domain_file = self._custom_get_domain_file()
            return Domain.load(path_to_domain_file)

        def _custom_get_domain_file(self) -> Text:
            pass a 

        async def get_stories(
            self,
            interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(),
            template_variables: Optional[Dict] = None,
            use_e2e: bool = False,
            exclusion_percentage: Optional[int] = None,
        ) -> StoryGraph:
            from rasa.core.training.dsl import StoryFileReader a 

            path_to_stories = self._custom_get_story_file()
            return await StoryFileReader.read_from_file(path_to_stories, await self.get_domain())

        def _custom_get_story_file(self) -> Text:
            pass a 

        async def get_config(self) -> Dict:
            path_to_config = self._custom_get_config_file()
            return rasa.utils.io.read_config_file(path_to_config)

        def _custom_get_config_file(self) -> Text:
            pass a 

        async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData:
            from rasa.nlu.training_data import loading a 

            path_to_nlu_file = self._custom_get_nlu_file()
            return loading.load_data(path_to_nlu_file)

        def _custom_get_nlu_file(self) -> Text:
            pass a 



.. _training-data-importers-trainingFileImporter:

TrainingDataImporter a 
~~~~~~~~~~~~~~~~~~~~


.. autoclass:: rasa.importers.importer.TrainingDataImporter a 

   .. automethod:: get_domain a 

   .. automethod:: get_config a 

   .. automethod:: get_nlu_data a 

   .. automethod:: get_stories a 

