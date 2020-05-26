:desc: Handle Rasa models on premise or in your private cloud for a
       GDPR-compliant intent recognition and entity extraction. a
 a
.. _cloud-storage: a
 a
Cloud Storage a
============= a
 a
.. edit-link:: a
 a
Rasa supports using `S3 <https://aws.amazon.com/s3/>`_ , a
`GCS <https://cloud.google.com/storage/>`_ and `Azure Storage <https://azure.microsoft.com/services/storage/>`_ to save your models. a
 a
* Amazon S3 Storage a
    S3 is supported using the ``boto3`` module which you can a
    install with ``pip install boto3``. a
 a
    Start the Rasa server with ``remote-storage`` option set to a
    ``aws``. Get your S3 credentials and set the following a
    environment variables: a
 a
    - ``AWS_SECRET_ACCESS_KEY`` a
    - ``AWS_ACCESS_KEY_ID`` a
    - ``AWS_DEFAULT_REGION`` a
    - ``BUCKET_NAME`` a
    - ``AWS_ENDPOINT_URL`` a
 a
    If there is no bucket with the name ``BUCKET_NAME``, Rasa will create it. a
 a
* Google Cloud Storage a
    GCS is supported using the ``google-cloud-storage`` package, a
    which you can install with ``pip install google-cloud-storage``. a
 a
    Start the Rasa server with ``remote-storage`` option set to ``gcs``. a
 a
    When running on google app engine and compute engine, the auth a
    credentials are already set up. For running locally or elsewhere, a
    checkout their a
    `client repo <https://github.com/GoogleCloudPlatform/python-docs-samples/tree/master/storage/cloud-client#authentication>`_ a
    for details on setting up authentication. It involves creating a
    a service account key file from google cloud console, a
    and setting the ``GOOGLE_APPLICATION_CREDENTIALS`` environment a
    variable to the path of that key file. a
 a
* Azure Storage a
    Azure is supported using the legacy ``azure-storage-blob`` package (v 2.1.0), a
    which you can install with ``pip install -I azure-storage-blob==2.1.0``. a
 a
    Start the Rasa server with ``remote-storage`` option set to ``azure``. a
 a
    The following environment variables must be set: a
 a
    - ``AZURE_CONTAINER`` a
    - ``AZURE_ACCOUNT_NAME`` a
    - ``AZURE_ACCOUNT_KEY`` a
 a
    If there is no container with the name ``AZURE_CONTAINER``, Rasa will create it. a
 a
Models are gzipped before they are saved in the cloud. The gzipped file naming convention a
is `{MODEL_NAME}.tar.gz` and it is stored in the root folder of the storage service. a
Currently, you are not able to manually specify the path on the cloud storage. a
 a
If storing trained models, Rasa will gzip the new model and upload it to the container. If retrieving/loading models a
from the cloud storage, Rasa will download the gzipped model locally and extract the contents to a temporary directory. a
 a