:desc: Read more about changes between major versions of our open source a
       NLP engine and how to migrate from one version to another. a
 a
.. _old-nlu-migration-guide: a
 a
Migration Guide a
=============== a
This page contains information about changes between major versions and a
how you can migrate from one version to another. a
 a
0.14.x to 0.15.0 a
---------------- a
 a
.. warning:: a
 a
  This is a release **breaking backwards compatibility**. a
  Unfortunately, it is not possible to load a
  previously trained models (as the stored file names have changed as a
  well as the configuration and metadata). Please make sure to retrain a
  a model before trying to use it with this improved version. a
 a
model configuration a
~~~~~~~~~~~~~~~~~~~ a
- The standard pipelines have been renamed. ``spacy_sklearn`` is now a
  ``pretrained_embeddings_spacy`` and ``tensorflow_embedding`` is now a
  ``supervised_embeddings``. a
- Components names used for nlu config have been changed. a
  Use component class name in nlu config file. a
 a
custom components a
~~~~~~~~~~~~~~~~~ a
- The signature of Component's methods have been changed: a
 a
  - ``load(...)``, ``create(...)`` and ``cache_key(...)`` methods a
    additionally take component's meta/config dicts a
  - ``persist(...)`` method additionally takes file name prefix a
    Change your custom components accordingly. a
 a
function names a
~~~~~~~~~~~~~~ a
- ``rasa_nlu.evaluate`` was renamed to ``rasa_nlu.test`` a
- ``rasa_nlu.test.run_cv_evaluation`` was renamed to a
  ``rasa_nlu.test.cross_validate`` a
- ``rasa_nlu.train.do_train()`` was renamed to to ``rasa_nlu.train.train()`` a
 a
0.13.x to 0.14.0 a
---------------- a
- ``/config`` endpoint removed, when training a new model, the user should a
  always post the configuration as part of the request instead of relying a
  on the servers config. a
- ``ner_duckling`` support has been removed. Use ``DucklingHTTPExtractor`` a
  instead. More info about ``DucklingHTTPExtractor`` can be found at a
  https://rasa.com/docs/nlu/components/#ner-duckling-http. a
 a
0.13.x to 0.13.3 a
---------------- a
- ``rasa_nlu.server`` has to  be supplied with a ``yml`` file defining the a
  model endpoint from which to retrieve training data. The file location has a
  be passed with the ``--endpoints`` argument, e.g. a
  ``rasa run --endpoints endpoints.yml`` a
  ``endpoints.yml`` needs to contain the ``model`` key a
  with a ``url`` and an optional ``token``. Here's an example: a
 a
  .. code-block:: yaml a
 a
    model: a
      url: http://my_model_server.com/models/default/nlu/tags/latest a
      token: my_model_server_token a
 a
  .. note:: a
 a
    If you configure ``rasa.nlu.server`` to pull models from a remote server, a
    the default project name will be used. It is defined a
    ``RasaNLUModelConfig.DEFAULT_PROJECT_NAME``. a
 a
 a
- ``rasa.nlu.train`` can also be run with the ``--endpoints`` argument a
  if you want to pull training data from a URL. Alternatively, the a
  current ``--url`` syntax is still supported. a
 a
  .. code-block:: yaml a
 a
    data: a
      url: http://my_data_server.com/projects/default/data a
      token: my_data_server_token a
 a
  .. note:: a
 a
    Your endpoint file may contain entries for both ``model`` and ``data``. a
    ``rasa.nlu.server`` and ``rasa.nlu.train`` will pick the relevant entry. a
 a
- If you directly access the ``DataRouter`` class or ``rasa.nlu.train``'s a
  ``do_train()`` method, you can directly create instances of a
  ``EndpointConfig`` without creating a ``yml`` file. Example: a
 a
  .. code-block:: python a
 a
    from rasa.nlu.utils import EndpointConfig a
    from rasa.nlu.data_router import DataRouter a
 a
    model_endpoint = EndpointConfig( a
        url="http://my_model_server.com/models/default/nlu/tags/latest", a
        token="my_model_server_token" a
    ) a
 a
    interpreter = DataRouter("projects", model_server=model_endpoint) a
 a
 a
0.12.x to 0.13.0 a
---------------- a
 a
.. warning:: a
 a
  This is a release **breaking backwards compatibility**. a
  Unfortunately, it is not possible to load previously trained models as a
  the parameters for the tensorflow and CRF models changed. a
 a
CRF model configuration a
~~~~~~~~~~~~~~~~~~~~~~~ a
 a
The feature names for the features of the entity CRF have changed: a
 a
+------------------+------------------+ a
| old feature name | new feature name | a
+==================+==================+ a
| pre2             | prefix2          | a
+------------------+------------------+ a
| pre5             | prefix5          | a
+------------------+------------------+ a
| word2            | suffix2          | a
+------------------+------------------+ a
| word3            | suffix3          | a
+------------------+------------------+ a
| word5            | suffix5          | a
+------------------+------------------+ a
 a
Please change these keys in your pipeline configuration of the ``CRFEntityExtractor`` a
components ``features`` attribute if you use them. a
 a
0.11.x to 0.12.0 a
---------------- a
 a
.. warning:: a
 a
  This is a release **breaking backwards compatibility**. a
  Unfortunately, it is not possible to load a
  previously trained models (as the stored file formats have changed as a
  well as the configuration and metadata). Please make sure to retrain a
  a model before trying to use it with this improved version. a
 a
model configuration a
~~~~~~~~~~~~~~~~~~~ a
We have split the configuration in a model configuration and parameters used a
to configure the server, train, and evaluate scripts. The configuration a
file now only contains the ``pipeline`` as well as the ``language`` a
parameters. Example: a
 a
  .. code-block:: yaml a
 a
      langauge: "en" a
 a
      pipeline: a
      - name: "SpacyNLP" a
        model: "en"               # parameter of the spacy component a
      - name: "EntitySynonymMapper" a
 a
 a
All other parameters have either been moved to the scripts a
for training, :ref:`serving models <running-the-server>`, or put into the a
:ref:`pipeline configuration <components>`. a
 a
persistors: a
~~~~~~~~~~~ a
- renamed ``AWS_REGION`` to ``AWS_DEFAULT_REGION`` a
- always make sure to specify the bucket using env ``BUCKET_NAME`` a
- are now configured solely over environment variables a
 a
0.9.x to 0.10.0 a
--------------- a
- We introduced a new concept called a ``project``. You can have multiple versions a
  of a model trained for a project. E.g. you can train an initial model and a
  add more training data and retrain that project. This will result in a new a
  model version for the same project. This allows you to, allways request a
  the latest model version from the http server and makes the model handling a
  more structured. a
- If you want to reuse trained models you need to move them in a directory named a
  after the project. E.g. if you already got a trained model in directory ``my_root/model_20170628-002704`` a
  you need to move that to ``my_root/my_project/model_20170628-002704``. Your a
  new projects name will be ``my_project`` and you can query the model using the a
  http server using ``curl http://localhost:5000/parse?q=hello%20there&project=my_project`` a
- Docs moved to https://rasahq.github.io/rasa_nlu/ a
- Renamed ``name`` parameter to ``project``. This means for training requests you now need to pass the ``project parameter a
  instead of ``name``, e.g. ``POST /train?project=my_project_name`` with the body of the a
  request containing the training data a
- Adapted remote cloud storages to support projects. This is a backwards incompatible change, a
  and unfortunately you need to retrain uploaded models and reupload them. a
 a
0.8.x to 0.9.x a
--------------- a
- add ``SpacyTokenizer`` to trained spacy_sklearn models metadata (right after the ``SpacyNLP``). alternative is to retrain the model a
 a
0.7.x to 0.8.x a
--------------- a
 a
- The training and loading capability for the spacy entity extraction was dropped in favor of the new CRF extractor. That means models need to be retrained using the crf extractor. a
 a
- The parameter and configuration value name of ``backend`` changed to ``pipeline``. a
 a
- There have been changes to the model metadata format. You can either retrain your models or change the stored a
  metadata.json: a
 a
    - rename ``language_name`` to ``language`` a
    - rename ``backend`` to ``pipeline`` a
    - for mitie models you need to replace ``feature_extractor`` with ``mitie_feature_extractor_fingerprint``. a
      That fingerprint depends on the language you are using, for ``en`` it a
      is ``"mitie_feature_extractor_fingerprint": 10023965992282753551``. a
 a
0.6.x to 0.7.x a
-------------- a
 a
- The parameter and configuration value name of ``server_model_dir`` changed to ``server_model_dirs``. a
 a
- The parameter and configuration value name of ``write`` changed to ``response_log``. It now configures the a
  *directory* where the logs should be written to (not a file!) a
 a
- The model metadata format has changed. All paths are now relative with respect to the ``path`` specified in the a
  configuration during training and loading. If you want to run models that are trained with a a
  version prev to 0.7 you need to adapt the paths manually in ``metadata.json`` from a
 a
  .. code-block:: json a
 a
      { a
          "trained_at": "20170304-191111", a
          "intent_classifier": "model_XXXX_YYYY_ZZZZ/intent_classifier.pkl", a
          "training_data": "model_XXXX_YYYY_ZZZZ/training_data.json", a
          "language_name": "en", a
          "entity_extractor": "model_XXXX_YYYY_ZZZZ/ner", a
          "feature_extractor": null, a
          "backend": "spacy_sklearn" a
      } a
 a
  to something along the lines of this (making all paths relative to the models base dir, which is ``model_XXXX_YYYY_ZZZZ/``): a
 a
  .. code-block:: json a
 a
      { a
          "trained_at": "20170304-191111", a
          "intent_classifier": "intent_classifier.pkl", a
          "training_data": "training_data.json", a
          "language_name": "en", a
          "entity_synonyms": null, a
          "entity_extractor": "ner", a
          "feature_extractor": null, a
          "backend": "spacy_sklearn" a
      } a
 a