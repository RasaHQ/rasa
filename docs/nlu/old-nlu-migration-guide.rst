:desc: Read more about changes between major versions of our open source
       NLP engine and how to migrate from one version to another.

.. _old-nlu-migration-guide:

Migration Guide
===============
This page contains information about changes between major versions and
how you can migrate from one version to another.

0.14.x to 0.15.0
----------------

.. warning::

  This is a release **breaking backwards compatibility**.
  Unfortunately, it is not possible to load
  previously trained models (as the stored file names have changed as
  well as the configuration and metadata). Please make sure to retrain
  a model before trying to use it with this improved version.

model configuration
~~~~~~~~~~~~~~~~~~~
- The standard pipelines have been renamed. ``spacy_sklearn`` is now
  ``pretrained_embeddings_spacy`` and ``tensorflow_embedding`` is now
  ``supervised_embeddings``.
- Components names used for nlu config have been changed.
  Use component class name in nlu config file.

custom components
~~~~~~~~~~~~~~~~~
- The signature of Component's methods have been changed:

  - ``load(...)``, ``create(...)`` and ``cache_key(...)`` methods
    additionally take component's meta/config dicts
  - ``persist(...)`` method additionally takes file name prefix
    Change your custom components accordingly.

function names
~~~~~~~~~~~~~~
- ``rasa_nlu.evaluate`` was renamed to ``rasa_nlu.test``
- ``rasa_nlu.test.run_cv_evaluation`` was renamed to
  ``rasa_nlu.test.cross_validate``
- ``rasa_nlu.train.do_train()`` was renamed to to ``rasa_nlu.train.train()``

0.13.x to 0.14.0
----------------
- ``/config`` endpoint removed, when training a new model, the user should
  always post the configuration as part of the request instead of relying
  on the servers config.
- ``ner_duckling`` support has been removed. Use ``DucklingHTTPExtractor``
  instead. More info about ``DucklingHTTPExtractor`` can be found at
  https://rasa.com/docs/nlu/components/#ner-duckling-http.

0.13.x to 0.13.3
----------------
- ``rasa_nlu.server`` has to  be supplied with a ``yml`` file defining the
  model endpoint from which to retrieve training data. The file location has
  be passed with the ``--endpoints`` argument, e.g.
  ``rasa run --endpoints endpoints.yml``
  ``endpoints.yml`` needs to contain the ``model`` key
  with a ``url`` and an optional ``token``. Here's an example:

  .. code-block:: yaml

    model:
      url: http://my_model_server.com/models/default/nlu/tags/latest
      token: my_model_server_token

  .. note::

    If you configure ``rasa.nlu.server`` to pull models from a remote server,
    the default project name will be used. It is defined
    ``RasaNLUModelConfig.DEFAULT_PROJECT_NAME``.


- ``rasa.nlu.train`` can also be run with the ``--endpoints`` argument
  if you want to pull training data from a URL. Alternatively, the
  current ``--url`` syntax is still supported.

  .. code-block:: yaml

    data:
      url: http://my_data_server.com/projects/default/data
      token: my_data_server_token

  .. note::

    Your endpoint file may contain entries for both ``model`` and ``data``.
    ``rasa.nlu.server`` and ``rasa.nlu.train`` will pick the relevant entry.

- If you directly access the ``DataRouter`` class or ``rasa.nlu.train``'s
  ``do_train()`` method, you can directly create instances of
  ``EndpointConfig`` without creating a ``yml`` file. Example:

  .. code-block:: python

    from rasa.nlu.utils import EndpointConfig
    from rasa.nlu.data_router import DataRouter

    model_endpoint = EndpointConfig(
        url="http://my_model_server.com/models/default/nlu/tags/latest",
        token="my_model_server_token"
    )

    interpreter = DataRouter("projects", model_server=model_endpoint)


0.12.x to 0.13.0
----------------

.. warning::

  This is a release **breaking backwards compatibility**.
  Unfortunately, it is not possible to load previously trained models as
  the parameters for the tensorflow and CRF models changed.

CRF model configuration
~~~~~~~~~~~~~~~~~~~~~~~

The feature names for the features of the entity CRF have changed:

+------------------+------------------+
| old feature name | new feature name |
+==================+==================+
| pre2             | prefix2          |
+------------------+------------------+
| pre5             | prefix5          |
+------------------+------------------+
| word2            | suffix2          |
+------------------+------------------+
| word3            | suffix3          |
+------------------+------------------+
| word5            | suffix5          |
+------------------+------------------+

Please change these keys in your pipeline configuration of the ``CRFEntityExtractor``
components ``features`` attribute if you use them.

0.11.x to 0.12.0
----------------

.. warning::

  This is a release **breaking backwards compatibility**.
  Unfortunately, it is not possible to load
  previously trained models (as the stored file formats have changed as
  well as the configuration and metadata). Please make sure to retrain
  a model before trying to use it with this improved version.

model configuration
~~~~~~~~~~~~~~~~~~~
We have split the configuration in a model configuration and parameters used
to configure the server, train, and evaluate scripts. The configuration
file now only contains the ``pipeline`` as well as the ``language``
parameters. Example:

  .. code-block:: yaml

      langauge: "en"

      pipeline:
      - name: "SpacyNLP"
        model: "en"               # parameter of the spacy component
      - name: "EntitySynonymMapper"


All other parameters have either been moved to the scripts
for training, :ref:`serving models <running-the-server>`, or put into the
:ref:`pipeline configuration <components>`.

persistors:
~~~~~~~~~~~
- renamed ``AWS_REGION`` to ``AWS_DEFAULT_REGION``
- always make sure to specify the bucket using env ``BUCKET_NAME``
- are now configured solely over environment variables

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
- add ``SpacyTokenizer`` to trained spacy_sklearn models metadata (right after the ``SpacyNLP``). alternative is to retrain the model

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
