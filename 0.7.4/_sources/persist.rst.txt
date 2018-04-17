.. _section_persistence:

Model Persistence
=================


rasa NLU supports using `S3 <https://aws.amazon.com/s3/>`_ to save your models, using the ``boto3``
module which you can install with ``pip install boto3``

Get your S3 credentials and set the following environment variables:

- ``AWS_SECRET_ACCESS_KEY``
- ``AWS_ACCESS_KEY_ID``
- ``AWS_REGION``
- ``BUCKET_NAME``

If there is no bucket with the name ``$BUCKET_NAME`` rasa will create it. 
Models are gzipped before saving to S3. 

If you run the rasa NLU server with a ``server_model_dirs`` which does not exist and ``BUCKET_NAME`` is set, rasa will attempt to fetch a matching zip from your S3 bucket.
E.g. if you have ``server_model_dirs = ./data/model_20161111-180000`` rasa will look for a file named ``model_20161111-180000.tar.gz`` in your bucket, unzip it and load the model.
