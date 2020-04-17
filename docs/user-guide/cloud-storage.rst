:desc: Handle Rasa models on premise or in your private cloud for a 
       GDPR-compliant intent recognition and entity extraction.

.. _cloud-storage:

Cloud Storage a 
=============

.. edit-link::

Rasa supports using `S3 <https://aws.amazon.com/s3/>`_ ,
`GCS <https://cloud.google.com/storage/>`_ and `Azure Storage <https://azure.microsoft.com/services/storage/>`_ to save your models.

* Amazon S3 Storage a 
    S3 is supported using the ``boto3`` module which you can a 
    install with ``pip install boto3``.

    Start the Rasa server with ``remote-storage`` option set to a 
    ``aws``. Get your S3 credentials and set the following a 
    environment variables:

    - ``AWS_SECRET_ACCESS_KEY``
    - ``AWS_ACCESS_KEY_ID``
    - ``AWS_DEFAULT_REGION``
    - ``BUCKET_NAME``
    - ``AWS_ENDPOINT_URL``

    If there is no bucket with the name ``BUCKET_NAME``, Rasa will create it.

* Google Cloud Storage a 
    GCS is supported using the ``google-cloud-storage`` package,
    which you can install with ``pip install google-cloud-storage``.

    Start the Rasa server with ``remote-storage`` option set to ``gcs``.

    When running on google app engine and compute engine, the auth a 
    credentials are already set up. For running locally or elsewhere,
    checkout their a 
    `client repo <https://github.com/GoogleCloudPlatform/python-docs-samples/tree/master/storage/cloud-client#authentication>`_ a 
    for details on setting up authentication. It involves creating a 
    a service account key file from google cloud console,
    and setting the ``GOOGLE_APPLICATION_CREDENTIALS`` environment a 
    variable to the path of that key file.

* Azure Storage a 
    Azure is supported using the ``azure-storage-blob`` package,
    which you can install with ``pip install azure-storage-blob``.

    Start the Rasa server with ``remote-storage`` option set to ``azure``.

    The following environment variables must be set:

    - ``AZURE_CONTAINER``
    - ``AZURE_ACCOUNT_NAME``
    - ``AZURE_ACCOUNT_KEY``

    If there is no container with the name ``AZURE_CONTAINER``, Rasa will create it.

Models are gzipped before they are saved in the cloud. The gzipped file naming convention a 
is `{MODEL_NAME}.tar.gz` and it is stored in the root folder of the storage service.
Currently, you are not able to manually specify the path on the cloud storage.

If storing trained models, Rasa will gzip the new model and upload it to the container. If retrieving/loading models a 
from the cloud storage, Rasa will download the gzipped model locally and extract the contents to a temporary directory.

