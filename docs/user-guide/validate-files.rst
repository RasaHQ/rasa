:desc: Check the domain, stories and intent files for possible errors.

.. _validate-files:

Validate Data
=============


Test Domain and Data Files for Mistakes
---------------------------------------

To verify if there are any mistakes in your domain file, NLU data, or story data, run the validate script.
You can run it with the following command:

.. code-block:: bash

  rasa data validate

The script above runs all the validations on your files. Here is the list of options to
the script:

.. program-output:: rasa data validate --help

You can also run these validations through the Python API by importing the `Validator` class,
which has the following methods:

**from_importer():** Creates the instance from Rasa File Importer.

**verify_intents():** Checks if intents listed in domain file are consistent with the NLU data.

**verify_intents_in_stories():** Verification for intents in the stories, to check if they are valid.

**verify_utterances():** Checks domain file for consistency between utterance templates and utterances listed under
actions.

**verify_utterances_in_stories():** Verification for utterances in stories, to check if they are valid.

**verify_all():** Runs all verifications above.

To use these functions it is necessary to create a `Validator` object and initialize the logger. See the following code:

.. code-block:: python

  import logging
  import asyncio
  from rasa.utils.io import configure_colored_logging
  from rasa.core.validator import Validator
  from rasa.importers.rasa import RasaFileImporter

  configure_colored_logging('DEBUG')

  logger = logging.getLogger(__name__)
  loop = asyncio.get_event_loop()

  importer = RasaFileImporter(
    domain_path="domain.yml",
    training_data_paths=["data/nlu_data.md", "data/stories.md"],
  )

  validator = loop.run_until_complete(Validator.from_importer(importer))

  validator.verify_all()
