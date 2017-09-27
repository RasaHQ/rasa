Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

0.9.x to 0.10.0
---------------
- We introduced a new concept called a ``project``. You can have multiple versions
  of a model trained for a project. E.g. you can train an initial model and
  add more training data and retrain that project. This will result in a new
  model version for the same project. This allows you to, allways request
  the latest model version from the http server and makes the model handling
  more structured.
- If you want to reuse trained models you need to move them in a directory named
  after the project. E.g. if you already got a trained model in directory ``my_root/model_20170628-002704``
  you need to move that to ``my_root/my_project/model_20170628-002704``. Your
  new projects name will be ``my_project`` and you can query the model using the
  http server using ``curl http://localhost:5000/parse?q=hello%20there&project=my_project``
- Docs moved to https://rasahq.github.io/rasa_nlu/
- Renamed ``name`` parameter to ``project``. This means for training requests you now need to pass the ``project parameter
  instead of ``name``, e.g. ``POST /train?project=my_project_name`` with the body of the
  request containing the training data
- Adapted remote cloud storages to support projects. This is a backwards incompatible change,
  and unfortunately you need to retrain uploaded models and reupload them.

0.8.x to 0.9.x
---------------
- add ``tokenizer_spacy`` to trained spacy_sklearn models metadata (right after the ``nlp_spacy``). alternative is to retrain the model

0.7.x to 0.8.x
---------------

- The training and loading capability for the spacy entity extraction was dropped in favor of the new CRF extractor. That means models need to be retrained using the crf extractor.

- The parameter and configuration value name of ``backend`` changed to ``pipeline``.

- There have been changes to the model metadata format. You can either retrain your models or change the stored
  metadata.json:

    - rename ``language_name`` to ``language``
    - rename ``backend`` to ``pipeline``
    - for mitie models you need to replace ``feature_extractor`` with ``mitie_feature_extractor_fingerprint``.
      That fingerprint depends on the language you are using, for ``en`` it
      is ``"mitie_feature_extractor_fingerprint": 10023965992282753551``.

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
