:desc: Check your domain, stories and intent files for possible errors. a
 a
.. _validate-files: a
 a
Validate Data a
============= a
 a
.. edit-link:: a
 a
 a
Test Domain and Data Files for Mistakes a
--------------------------------------- a
 a
To verify if there are any mistakes in your domain file, NLU data, or story data, run the validate script. a
You can run it with the following command: a
 a
.. code-block:: bash a
 a
  rasa data validate a
 a
The script above runs all the validations on your files, except for story structure validation, a
which is omitted unless you provide the ``--max-history`` argument. Here is the list of options to a
the script: a
 a
.. program-output:: rasa data validate --help a
 a
By default the validator searches only for errors in the data (e.g. the same a
example being listed as an example for two intents), but does not report other a
minor issues (such as unused intents, utterances that are not listed as a
actions). To also report the later use the ``-debug`` flag. a
 a
You can also run these validations through the Python API by importing the `Validator` class, a
which has the following methods: a
 a
**from_files():** Creates the instance from string paths to the necessary files. a
 a
**verify_intents():** Checks if intents listed in domain file are consistent with the NLU data. a
 a
**verify_example_repetition_in_intents():** Checks if there is no duplicated data among distinct intents at NLU data. a
 a
**verify_intents_in_stories():** Verification for intents in the stories, to check if they are valid. a
 a
**verify_utterances():** Checks domain file for consistency between responses listed in the `responses` section  a
and the utterance actions you have defined. a
 a
**verify_utterances_in_stories():** Verification for utterances in stories, to check if they are valid. a
 a
**verify_all():** Runs all verifications above. a
 a
**verify_domain_validity():** Check if domain is valid. a
 a
To use these functions it is necessary to create a `Validator` object and initialize the logger. See the following code: a
 a
.. code-block:: python a
 a
  import logging a
  from rasa import utils a
  from rasa.core.validator import Validator a
 a
  logger = logging.getLogger(__name__) a
 a
  utils.configure_colored_logging('DEBUG') a
 a
  validator = Validator.from_files(domain_file='domain.yml', a
                                   nlu_data='data/nlu_data.md', a
                                   stories='data/stories.md') a
 a
  validator.verify_all() a
 a
.. _test-story-files-for-conflicts: a
 a
Test Story Files for Conflicts a
------------------------------ a
 a
In addition to the default tests described above, you can also do a more in-depth structural test of your stories. a
In particular, you can test if your stories are inconsistent, i.e. if different bot actions follow from the same dialogue history. a
If this is not the case, then Rasa cannot learn the correct behaviour. a
 a
Take, for example, the following two stories: a
 a
.. code-block:: md a
 a
  ## Story 1 a
  * greet a
    - utter_greet a
  * inform_happy a
    - utter_happy a
    - utter_goodbye a
 a
  ## Story 2 a
  * greet a
    - utter_greet a
  * inform_happy a
    - utter_goodbye a
 a
These two stories are inconsistent, because Rasa doesn't know if it should predict ``utter_happy`` or ``utter_goodbye`` a
after ``inform_happy``, as there is nothing that would distinguish the dialogue states at ``inform_happy`` in the two  a
stories and the subsequent actions are different in Story 1 and Story 2. a
 a
This conflict can be automatically identified with our story structure validation tool. a
To do this, use ``rasa data validate`` in the command line, as follows: a
 a
.. code-block:: bash a
 a
  rasa data validate stories --max-history 3 a
  > 2019-12-09 09:32:13 INFO     rasa.core.validator  - Story structure validation... a
  > 2019-12-09 09:32:13 INFO     rasa.core.validator  - Assuming max_history = 3 a
  >   Processed Story Blocks: 100% 2/2 [00:00<00:00, 3237.59it/s, # trackers=1] a
  > 2019-12-09 09:32:13 WARNING  rasa.core.validator  - CONFLICT after intent 'inform_happy': a
  >   utter_goodbye predicted in 'Story 2' a
  >   utter_happy predicted in 'Story 1' a
 a
Here we specify a ``max-history`` value of 3. a
This means, that 3 events (user messages / bot actions) are taken into account for action predictions, but the particular setting does not matter for this example, because regardless of how long of a history you take into account, the conflict always exists. a
 a
.. warning:: a
 a
    The ``rasa data validate stories`` script assumes that all your **story names are unique**. a
    If your stories are in the Markdown format, you may find duplicate names with a command like a
    ``grep -h "##" data/*.md | uniq -c | grep "^[^1]"``. a
 a