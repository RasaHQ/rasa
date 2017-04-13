from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object
import os
import shutil
import tarfile
import io

import boto3
import botocore
from typing import Text


class Persistor(object):
    """Store models on S3 and fetch them when needed instead of storing them on the local disk."""

    def __init__(self, data_dir, aws_region, bucket_name):
        self.data_dir = data_dir
        self.s3 = boto3.resource('s3', region_name=aws_region)
        self.bucket_name = bucket_name
        try:
            self.s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': aws_region})
        except botocore.exceptions.ClientError as e:
            pass  # bucket already exists
        self.bucket = self.s3.Bucket(bucket_name)

    def send_tar_to_s3(self, target_dir):
        # type: (Text) -> None
        """Uploads a model persisted in the `target_dir` to s3."""

        if not os.path.isdir(target_dir):
            raise ValueError("Target directory '{}' not found.".format(target_dir))

        base_name = os.path.basename(target_dir)
        base_dir = os.path.dirname(target_dir)
        tarname = shutil.make_archive(base_name, 'gztar', root_dir=base_dir, base_dir=base_name)
        filekey = os.path.basename(tarname)
        self.s3.Object(self.bucket_name, filekey).put(Body=open(tarname, 'rb'))

    def fetch_and_extract(self, filename):
        # type: (Text) -> None
        """Downloads a model that has previously been persisted to s3."""

        with io.open(filename, 'wb') as f:
            self.bucket.download_fileobj(filename, f)
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(self.data_dir)
