import io
from rasa.nlu.utils import zip_folder

import mock
import responses

from rasa.nlu.components import ComponentBuilder
from rasa.nlu.model_loader import NLUModel, load_from_server
from rasa.utils.endpoints import EndpointConfig


@responses.activate
async def test_project_with_model_server(trained_nlu_model):
    fingerprint = "somehash"
    model_endpoint = EndpointConfig("http://server.com/models/nlu/tags/latest")

    zip_path = zip_folder(trained_nlu_model)

    # mock a response that returns a zipped model
    with io.open(zip_path, "rb") as f:
        responses.add(
            responses.GET,
            model_endpoint.url,
            headers={"ETag": fingerprint, "filename": "my_model_xyz.zip"},
            body=f.read(),
            content_type="application/zip",
            stream=True,
        )
    nlu_model = await load_from_server(
        ComponentBuilder(use_cache=False), model_server=model_endpoint
    )
    assert nlu_model.fingerprint == fingerprint
