#!/usr/bin/env python

"""Download pretrained transformer models"""

import os
import pathlib
import requests

hf_transformers_models_fpath = pathlib.Path(__file__).parents[1] / "data/test/hf_transformers_models.txt"

is_ci_enabled = os.getenv('CI')
operating_system = os.getenv('OS')

HOME_DIR=os.getenv('HOME')
if operating_system == "Windows_NT" :
    HOME_DIR=f"{HOME_DIR}{os.getenv('HOMEDRIVE')}"

CACHE_DIR=pathlib.Path(HOME_DIR) / ".cache/torch/transformers"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    with open(hf_transformers_models_fpath) as fp:
        i = 0
        while True:
            url = fp.readline()
            cache_file = fp.readline()
            if not url or not cache_file:
                break
            num_models_to_skip = 1  # 4
            if is_ci_enabled and i > num_models_to_skip:
                print(url, CACHE_DIR / cache_file)
                r = requests.get(url)
                open(CACHE_DIR / cache_file , 'wb').write(r.content)

            i += 1
