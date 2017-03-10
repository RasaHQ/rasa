Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

0.6.x to 0.7.x
--------------

- The parameter and configuration value name of ``server_model_dir`` changed to ``server_model_dirs``.

- The parameter and configuration value name of ``write`` changed to ``response_log``. It now configures the
  *directory* where the logs should be written to (not a file!)

- The model metadata format has changed. All paths are now relative with respect to the ``path`` specified in the
  configuration during training and loading. If you want to run models that are trained with a
  version prev to 0.7 you need to adapt the paths manually in ``metadata.json`` from

  .. code-block:: json

      {
          "trained_at": "20170304-191111",
          "intent_classifier": "model_XXXX_YYYY_ZZZZ/intent_classifier.pkl",
          "training_data": "model_XXXX_YYYY_ZZZZ/training_data.json",
          "language_name": "en",
          "entity_extractor": "model_XXXX_YYYY_ZZZZ/ner",
          "feature_extractor": null,
          "backend": "spacy_sklearn"
      }

  to something along the lines of this (making all paths relative to the models base dir, which is ``model_XXXX_YYYY_ZZZZ/``):

  .. code-block:: json

      {
          "trained_at": "20170304-191111",
          "intent_classifier": "intent_classifier.pkl",
          "training_data": "training_data.json",
          "language_name": "en",
          "entity_synonyms": null,
          "entity_extractor": "ner",
          "feature_extractor": null,
          "backend": "spacy_sklearn"
      }
