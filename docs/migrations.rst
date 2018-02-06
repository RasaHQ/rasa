.. _migration:

Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

0.7.x to 0.8.0
--------------

- Credentials for the facebook connector changed. Instead of providing

  .. code-block:: yaml

      # OLD FORMAT
      verify: "rasa-bot"
      secret: "3e34709d01ea89032asdebfe5a74518"
      page-tokens:
        1730621093913654: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"

  you should now pass the configuration parameters like this:

  .. code-block:: yaml

      # NEW FORMAT
      verify: "rasa-bot"
      secret: "3e34709d01ea89032asdebfe5a74518"
      page-access-token: "EAAbHPa7H9rEBAAuFk4Q3gPKbDedQnx4djJJ1JmQ7CAqO4iJKrQcNT0wtD"

  As you can see, the new facebook connector only supports a single page. Same
  change happened to the in code arguments for the connector which should be
  changed to:

  .. code-block:: python

      from rasa_core.channels.facebook import FacebookInput

      FacebookInput(
            credentials.get("verify"),
            credentials.get("secret"),
            credentials.get("page-access-token"))

- Story file format changed from ``* _intent_greet[name=Rasa]``
  to ``* intent_greet{"name": "Rasa"}`` (old format is still supported but
  deprecated). Instead of writing

  .. code-block:: md

      ## story_07715946                     <!-- name of the story - just for debugging -->
      * _greet
         - action_ask_howcanhelp
      * _inform[location=rome,price=cheap]
         - action_on_it                     <!-- user utterance, in format _intent[entities] -->
         - action_ask_cuisine

  The new format looks like this:

  .. code-block:: md

      ## story_07715946                     <!-- name of the story - just for debugging -->
      * greet
         - action_ask_howcanhelp
      * inform{"location": "rome", "price": "cheap"}
         - action_on_it                     <!-- user utterance, in format _intent[entities] -->
         - action_ask_cuisine
