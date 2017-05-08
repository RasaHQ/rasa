.. _section_persistence:

Model Persistence
=================


rasa NLU supports using `S3 <https://aws.amazon.com/s3/>`_ , `GCS <https://cloud.google.com/storage/>`_  and `MongoDB <https://www.mongodb.com/>`_ to save your models.

* Amazon S3 Storage
    S3 is supported using the ``boto3`` module which you can install with ``pip install boto3``.

    Start the rasa NLU server with ``storage`` option set to ``aws``. Get your S3
    credentials and set the following environment variables:

    - ``AWS_SECRET_ACCESS_KEY``
    - ``AWS_ACCESS_KEY_ID``
    - ``AWS_REGION``
    - ``BUCKET_NAME``


* Google Cloud Storage
    GCS is supported using the ``google-cloud-storage`` package which you can install with ``pip install google-cloud-storage``

    Start the rasa NLU server with ``storage`` option set to ``gcs``.

    When running on google app engine and compute engine, the auth credentials are
    already set up. For running locally or elsewhere, checkout their `client repo <https://github.com/GoogleCloudPlatform/python-docs-samples/tree/master/storage/cloud-client#authentication>`_ for details on
    setting up authentication. It involves creating a service account key file from google cloud console, and setting the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to the path of that key file.

If there is no bucket with the name ``$BUCKET_NAME`` rasa will create it.
Models are gzipped before saving to cloud.

If you run the rasa NLU server with a ``server_model_dirs`` which does not exist and ``BUCKET_NAME`` is set, rasa will attempt to fetch a matching zip from your cloud storage bucket.
E.g. if you have ``server_model_dirs = ./data/model_20161111-180000`` rasa will look for a file named ``model_20161111-180000.tar.gz`` in your bucket, unzip it and load the model.


* MongoDB
    MongoDB is supported using the ``PyMongo`` module which you can install with ``pip install pymongo``.

    If you trained your model with storage set to ``mongodb`` it will store the files to the MongoDB as it is, creating a new document with the key ``model_name`` having the value of the model name (E.g.: ``model_20161111-180000``).

    Start the rasa NLU server with ``storage`` option set to ``mongodb`` and providing collection name as ``collection_name``.


    Then you run the rasa NLU server with a ``server_model_dirs`` providing model name (E.g.: ``model_20161111-180000``), rasa will attempt to fetch a matching document in the given MongoDB collection.
    If you have ``server_model_dirs = model_20161111-180000`` rasa will look for the document with ``model_name`` set to ``model_20161111-180000``. It will create necessary files and directories and place the models in the file system accordingly.