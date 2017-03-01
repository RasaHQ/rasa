import boto3
import botocore
import tarfile
import os
import shutil

import google
from google.cloud import storage
from google.cloud import exceptions


def get_persistor(config):
    p = None
    if config['storage'] == 'aws':
        p = AWSPersistor(config['path'], config['aws_region'], config['bucket_name'])
    elif config['storage'] == 'gcs':
        p = GCSPersistor(config['path'], config['bucket_name'])
    return p


class AWSPersistor(object):
    def __init__(self, data_dir, aws_region, bucket_name):
        self.data_dir = data_dir
        self.s3 = boto3.resource('s3', region_name=aws_region)
        self.bucket_name = bucket_name
        try:
            self.s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': aws_region})
        except botocore.exceptions.ClientError, e:
            pass  # bucket already exists
        self.bucket = self.s3.Bucket(bucket_name)

    def save_tar(self, target_dir):

        if not os.path.isdir(target_dir):
            raise ValueError('target_dir %r not found.' % target_dir)

        base_name = os.path.basename(target_dir)
        base_dir = os.path.dirname(target_dir)
        tarname = shutil.make_archive(base_name, 'gztar', root_dir=base_dir, base_dir=base_name)
        filekey = os.path.basename(tarname)
        self.s3.Object(self.bucket_name, filekey).put(Body=open(tarname, 'rb'))

    def fetch_and_extract(self, filename):
        with open(filename, 'wb') as f:
            self.bucket.download_fileobj(filename, f)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(self.data_dir)


class GCSPersistor(object):
    def __init__(self, data_dir, bucket_name):
        self.data_dir = data_dir
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()

        try:
            self.storage_client.create_bucket(bucket_name)
        except exceptions.Conflict, e:
            # bucket exists
            pass
        self.bucket = self.storage_client.bucket(bucket_name)

    def save_tar(self, target_dir):
        if not os.path.isdir(target_dir):
            raise ValueError('target_dir %r not found.' % target_dir)

        base_name = os.path.basename(target_dir)
        base_dir = os.path.dirname(target_dir)
        tarname = shutil.make_archive(base_name, 'gztar', root_dir=base_dir, base_dir=base_name)
        filekey = os.path.basename(tarname)
        blob = self.bucket.blob(filekey)
        blob.upload_from_filename(tarname)

    def fetch_and_extract(self, filename):
        blob = self.bucket.blob(filename)
        blob.download_to_filename(filename)

        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(self.data_dir)
