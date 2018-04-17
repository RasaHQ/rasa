.. _section_persistence:

Model Persistence
=================


rasa NLU supports using `S3 <https://aws.amazon.com/s3/>`_ and `GCS <https://cloud.google.com/storage/>`_ to save your models.

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
