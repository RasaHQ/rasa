#!/usr/bin/env python

"""Extract the TensorFlow version from poetry.lock. Used in e.g. github workflows."""

import pathlib
import sys
import toml

poetry_path = 'poetry.lock'

if __name__ == "__main__":
    if pathlib.Path(poetry_path).is_file():
      # Get the list of dict that contains all information about packages
      package_list = toml.load(poetry_path).get('package')
      # Extract tensorflow version
      tf_version = next(filter(lambda x: x.get('name') == 'tensorflow', package_list)).get('version')
      print(tf_version)
      sys.exit(0)

    else:
      print(f'File {poetry_path} does not exist.')
      sys.exit(1)
