:desc: Change the way Rasa imports training data by replacing the default importer or a
       writing your own importer. a
 a
.. _training-data-importers: a
 a
Training Data Importers a
======================= a
 a
.. edit-link:: a
 a
.. contents:: a
   :local: a
 a
By default, you can use command line arguments to specify where Rasa should look a
for training data on your disk. Rasa then loads any potential training files and uses a
them to train your assistant. a
 a
If needed, you can also customize `how` Rasa imports training data. a
Potential use cases for this might be: a
 a
- using a custom parser to load training data in other formats a
- using different approaches to collect training data (e.g. loading them from different resources) a
 a
You can instruct Rasa to load and use your custom importer by adding the section a
``importers`` to the Rasa configuration file and specifying the importer with its a
full class path: a
 a
.. code-block:: yaml a
 a
   importers: a
   - name: "module.CustomImporter" a
     parameter1: "value" a
     parameter2: "value2" a
   - name: "module.AnotherCustomImporter" a
 a
The ``name`` key is used to determine which importer should be loaded. Any extra a
parameters are passed as constructor arguments to the loaded importer. a
 a
.. note:: a
 a
   You can specify multiple importers. Rasa will automatically merge their results. a
 a
 a
RasaFileImporter (default) a
~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
By default Rasa uses the importer ``RasaFileImporter``. If you want to use it on its a
own, you don't have to specify anything in your configuration file. a
If you want to use it together with other importers, add it to your a
configuration file: a
 a
.. code-block:: yaml a
 a
   importers: a
   - name: "RasaFileImporter" a
 a
MultiProjectImporter (experimental) a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
.. warning:: a
 a
    This feature is currently experimental and might change or be removed in the future. a
    Please share your feedback on it in the `forum <https://forum.rasa.com>`_ to help a
    us making this feature ready for production. a
 a
With this importer you can build a contextual AI assistant by combining multiple a
reusable Rasa projects. a
You might, for example, handle chitchat with one project and greet your users with a
another. These projects can be developed in isolation, and then combined at train time a
to create your assistant. a
 a
An example directory structure could look like this: a
 a
.. code-block:: bash a
 a
    . a
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
 a
In this example the contextual AI assistant imports the ``ChitchatBot`` project which in turn a
imports the ``GreetBot`` project. Project imports are defined in the configuration files of a
each project. a
To instruct Rasa to use the ``MultiProjectImporter`` module, put this section in the config a
file of your root project: a
 a
.. code-block:: yaml a
 a
    importers: a
    - name: MultiProjectImporter a
 a
 a
Then specify which projects you want to import. a
In our example, the ``config.yml`` in the root project would look like this: a
 a
.. code-block:: yaml a
 a
    imports: a
    - projects/ChitchatBot a
 a
The configuration file of the ``ChitchatBot`` in turn references the ``GreetBot``: a
 a
.. code-block:: yaml a
 a
    imports: a
    - ../GreetBot a
 a
The ``GreetBot`` project does not specify further projects so the ``config.yml`` can be a
omitted. a
 a
Rasa uses relative paths from the referencing configuration file to import projects. a
These can be anywhere on your file system as long as the file access is permitted. a
 a
During the training process Rasa will import all required training files, combine a
them, and train a unified AI assistant. The merging of the training data happens during a
runtime, so no additional files with training data are created or visible. a
 a
.. note:: a
 a
    Rasa will use the policy and NLU pipeline configuration of the root project a
    directory during training. **Policy or NLU configurations of imported projects a
    will be ignored.** a
 a
.. note:: a
 a
    Equal intents, entities, slots, responses, actions and forms will be merged, a
    e.g. if two projects have training data for an intent ``greet``, a
    their training data will be combined. a
 a
Writing a Custom Importer a
~~~~~~~~~~~~~~~~~~~~~~~~~ a
If you are writing a custom importer, this importer has to implement the interface of a
:ref:`training-data-importers-trainingFileImporter`: a
 a
.. code-block:: python a
 a
    from typing import Optional, Text, Dict, List, Union a
 a
    import rasa a
    from rasa.core.domain import Domain a
    from rasa.core.interpreter import RegexInterpreter, NaturalLanguageInterpreter a
    from rasa.core.training.structures import StoryGraph a
    from rasa.importers.importer import TrainingDataImporter a
    from rasa.nlu.training_data import TrainingData a
 a
 a
    class MyImporter(TrainingDataImporter): a
        """Example implementation of a custom importer component.""" a
 a
        def __init__( a
            self, a
            config_file: Optional[Text] = None, a
            domain_path: Optional[Text] = None, a
            training_data_paths: Optional[Union[List[Text], Text]] = None, a
            **kwargs: Dict a
        ): a
            """Constructor of your custom file importer. a
 a
            Args: a
                config_file: Path to configuration file from command line arguments. a
                domain_path: Path to domain file from command line arguments. a
                training_data_paths: Path to training files from command line arguments. a
                **kwargs: Extra parameters passed through configuration in configuration file. a
            """ a
 a
            pass a
 a
        async def get_domain(self) -> Domain: a
            path_to_domain_file = self._custom_get_domain_file() a
            return Domain.load(path_to_domain_file) a
 a
        def _custom_get_domain_file(self) -> Text: a
            pass a
 a
        async def get_stories( a
            self, a
            interpreter: "NaturalLanguageInterpreter" = RegexInterpreter(), a
            template_variables: Optional[Dict] = None, a
            use_e2e: bool = False, a
            exclusion_percentage: Optional[int] = None, a
        ) -> StoryGraph: a
            from rasa.core.training.dsl import StoryFileReader a
 a
            path_to_stories = self._custom_get_story_file() a
            return await StoryFileReader.read_from_file(path_to_stories, await self.get_domain()) a
 a
        def _custom_get_story_file(self) -> Text: a
            pass a
 a
        async def get_config(self) -> Dict: a
            path_to_config = self._custom_get_config_file() a
            return rasa.utils.io.read_config_file(path_to_config) a
 a
        def _custom_get_config_file(self) -> Text: a
            pass a
 a
        async def get_nlu_data(self, language: Optional[Text] = "en") -> TrainingData: a
            from rasa.nlu.training_data import loading a
 a
            path_to_nlu_file = self._custom_get_nlu_file() a
            return loading.load_data(path_to_nlu_file) a
 a
        def _custom_get_nlu_file(self) -> Text: a
            pass a
 a
 a
 a
.. _training-data-importers-trainingFileImporter: a
 a
TrainingDataImporter a
~~~~~~~~~~~~~~~~~~~~ a
 a
 a
.. autoclass:: rasa.importers.importer.TrainingDataImporter a
 a
   .. automethod:: get_domain a
 a
   .. automethod:: get_config a
 a
   .. automethod:: get_nlu_data a
 a
   .. automethod:: get_stories a
 a