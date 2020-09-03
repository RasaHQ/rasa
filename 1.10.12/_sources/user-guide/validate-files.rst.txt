:desc: Check your domain, stories and intent files for possible errors.

.. _validate-files:

Validate Data
=============

.. edit-link::


Test Domain and Data Files for Mistakes
---------------------------------------

To verify if there are any mistakes in your domain file, NLU data, or story data, run the validate script.
You can run it with the following command:

.. code-block:: bash

  rasa data validate

The script above runs all the validations on your files, except for story structure validation,
which is omitted unless you provide the ``--max-history`` argument. Here is the list of options to
the script:

.. program-output:: rasa data validate --help

By default the validator searches only for errors in the data (e.g. the same
example being listed as an example for two intents), but does not report other
minor issues (such as unused intents, utterances that are not listed as
actions). To also report the later use the ``-debug`` flag.

You can also run these validations through the Python API by importing the `Validator` class,
which has the following methods:

**from_files():** Creates the instance from string paths to the necessary files.

**verify_intents():** Checks if intents listed in domain file are consistent with the NLU data.

**verify_example_repetition_in_intents():** Checks if there is no duplicated data among distinct intents at NLU data.

**verify_intents_in_stories():** Verification for intents in the stories, to check if they are valid.

**verify_utterances():** Checks domain file for consistency between responses listed in the `responses` section 
and the utterance actions you have defined.

**verify_utterances_in_stories():** Verification for utterances in stories, to check if they are valid.

**verify_all():** Runs all verifications above.

**verify_domain_validity():** Check if domain is valid.

To use these functions it is necessary to create a `Validator` object and initialize the logger. See the following code:

.. code-block:: python

  import logging
  from rasa import utils
  from rasa.core.validator import Validator

  logger = logging.getLogger(__name__)

  utils.configure_colored_logging('DEBUG')

  validator = Validator.from_files(domain_file='domain.yml',
                                   nlu_data='data/nlu_data.md',
                                   stories='data/stories.md')

  validator.verify_all()

.. _test-story-files-for-conflicts:

Test Story Files for Conflicts
------------------------------

In addition to the default tests described above, you can also do a more in-depth structural test of your stories.
In particular, you can test if your stories are inconsistent, i.e. if different bot actions follow from the same dialogue history.
If this is not the case, then Rasa cannot learn the correct behavior.

Take, for example, the following two stories:

.. code-block:: md

  ## Story 1
  * greet
    - utter_greet
  * inform_happy
    - utter_happy
    - utter_goodbye

  ## Story 2
  * greet
    - utter_greet
  * inform_happy
    - utter_goodbye

These two stories are inconsistent, because Rasa doesn't know if it should predict ``utter_happy`` or ``utter_goodbye``
after ``inform_happy``, as there is nothing that would distinguish the dialogue states at ``inform_happy`` in the two 
stories and the subsequent actions are different in Story 1 and Story 2.

This conflict can be automatically identified with our story structure validation tool.
To do this, use ``rasa data validate`` in the command line, as follows:

.. code-block:: bash

  rasa data validate stories --max-history 3
  > 2019-12-09 09:32:13 INFO     rasa.core.validator  - Story structure validation...
  > 2019-12-09 09:32:13 INFO     rasa.core.validator  - Assuming max_history = 3
  >   Processed Story Blocks: 100% 2/2 [00:00<00:00, 3237.59it/s, # trackers=1]
  > 2019-12-09 09:32:13 WARNING  rasa.core.validator  - CONFLICT after intent 'inform_happy':
  >   utter_goodbye predicted in 'Story 2'
  >   utter_happy predicted in 'Story 1'

Here we specify a ``max-history`` value of 3.
This means, that 3 events (user messages / bot actions) are taken into account for action predictions, but the particular setting does not matter for this example, because regardless of how long of a history you take into account, the conflict always exists.

.. warning::

    The ``rasa data validate stories`` script assumes that all your **story names are unique**.
    If your stories are in the Markdown format, you may find duplicate names with a command like
    ``grep -h "##" data/*.md | uniq -c | grep "^[^1]"``.
