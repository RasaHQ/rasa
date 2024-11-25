#!/usr/bin/env python
import os

from huggingface_hub import get_inference_endpoint

ENDPOINT_NAME = os.environ["ENDPOINT_NAME"]
ENDPOINT_NAMESPACE = os.getenv("ENDPOINT_NAMESPACE", "rasa")
HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]

if __name__ == "__main__":
    endpoint = (
        get_inference_endpoint(
            name=ENDPOINT_NAME, namespace=ENDPOINT_NAMESPACE, token=HUGGINGFACE_API_KEY
        )
        .resume()
        .wait()
    )
    assert endpoint.status == "running"
