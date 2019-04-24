import datetime
import logging
import os
import tempfile
import time
from threading import Lock, Thread
from typing import Optional, Text

import rasa.utils.io
from rasa import model
from rasa.model import get_latest_model

from rasa.nlu import utils
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.model import Interpreter, Metadata, MODEL_NAME_PREFIX
from rasa.nlu.utils import is_url
from rasa.utils.endpoints import EndpointConfig
from requests.exceptions import InvalidURL, RequestException


logger = logging.getLogger(__name__)


FALLBACK_MODEL_NAME = "fallback"

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes


def interpreter_for_model(component_builder, model_dir):
    metadata = Metadata.load(model_dir)
    return Interpreter.create(metadata, component_builder)


async def load_from_server(
    component_builder: ComponentBuilder,
    model_server: EndpointConfig,
    wait_time_between_pulls: Optional[int] = None,
) -> "NLUModel":
    """Load a persisted model from a server."""

    nlu_model = NLUModel.fallback_model(component_builder)

    await _update_model_from_server(model_server, nlu_model, component_builder)

    if wait_time_between_pulls:
        # continuously pull the model every `wait_time_between_pulls` seconds
        start_model_pulling_in_worker(
            component_builder, model_server, wait_time_between_pulls, nlu_model
        )

    return nlu_model


async def _update_model_from_server(
    model_server: EndpointConfig,
    nlu_model: "NLUModel",
    component_builder: ComponentBuilder,
) -> None:
    """Load a tar.gz Rasa NLU model from a URL and update the passed
    nlu model."""
    if not is_url(model_server.url):
        raise InvalidURL(model_server)

    model_directory = tempfile.mkdtemp()

    new_model_fingerprint, filename = await _pull_model_and_fingerprint(
        model_server, model_directory, nlu_model.fingerprint
    )

    if new_model_fingerprint:
        model_name = _get_remote_model_name(filename)
        nlu_model.fingerprint = new_model_fingerprint
        nlu_model.update_model(component_builder, model_directory, model_name)
    else:
        logger.debug("No new model found at URL '{}'".format(model_server.url))


def _get_remote_model_name(filename: Optional[Text]) -> Text:
    """Get the name to save a model under that was fetched from a
    remote server."""
    if filename is not None:  # use the filename header if present
        return filename.strip(".tar.gz")
    else:  # or else use a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return MODEL_NAME_PREFIX + timestamp


async def _pull_model_and_fingerprint(
    model_server: EndpointConfig, model_directory: Text, fingerprint: Optional[Text]
) -> (Optional[Text], Optional[Text]):
    """Queries the model server and returns a tuple of containing the
    response's <ETag> header which contains the model hash, and the
    <filename> header containing the model name."""
    header = {"If-None-Match": fingerprint}
    try:
        logger.debug("Requesting model from server {}...".format(model_server.url))
        response = await model_server.request(
            method="GET", headers=header, timeout=DEFAULT_REQUEST_TIMEOUT
        )
    except RequestException as e:
        logger.warning(
            "Tried to fetch model from server, but couldn't reach "
            "server. We'll retry later... Error: {}."
            "".format(e)
        )
        return None, None

    if response.status_code == 204:
        logger.debug(
            "Model server returned 204 status code, indicating "
            "that no new model is available. "
            "Current fingerprint: {}".format(fingerprint)
        )
        return response.headers.get("ETag"), response.headers.get("filename")
    elif response.status_code == 404:
        logger.debug(
            "Model server didn't find a model for that request. "
            "Probably no model trained with that name yet."
        )
        return None, None
    elif response.status_code != 200:
        logger.warning(
            "Tried to fetch model from server, but server response "
            "status code is {}. We'll retry later..."
            "".format(response.status_code)
        )
        return None, None

    rasa.utils.io.unarchive(await response.read(), model_directory)
    logger.debug("Unzipped model to '{}'".format(os.path.abspath(model_directory)))

    # get the new fingerprint and filename
    return response.headers.get("ETag"), response.headers.get("filename")


async def _run_model_pulling_worker(
    model_server: EndpointConfig,
    wait_time_between_pulls: int,
    nlu_model: "NLUModel",
    component_builder: ComponentBuilder,
) -> None:
    while True:
        await _update_model_from_server(model_server, nlu_model, component_builder)
        time.sleep(wait_time_between_pulls)


def start_model_pulling_in_worker(
    component_builder: ComponentBuilder,
    model_server: Optional[EndpointConfig],
    wait_time_between_pulls: int,
    nlu_model: "NLUModel",
) -> None:
    worker = Thread(
        target=_run_model_pulling_worker,
        args=(model_server, wait_time_between_pulls, nlu_model, component_builder),
    )
    worker.setDaemon(True)
    worker.start()


class NLUModel(object):
    def __init__(
        self,
        model_name: Text,
        interpreter: Interpreter,
        model_path: Optional[Text] = None,
        fingerprint: Optional[Text] = None,
    ):
        self.name = model_name
        self.path = model_path
        self.interpreter = interpreter
        self.fingerprint = fingerprint

        self._reader_lock = Lock()
        self._loader_lock = Lock()
        self._writer_lock = Lock()
        self._readers_count = 0

    def _begin_read(self):
        self._reader_lock.acquire()
        self._readers_count += 1
        if self._readers_count == 1:
            self._writer_lock.acquire()
        self._reader_lock.release()

    def _end_read(self):
        self._reader_lock.acquire()
        self._readers_count -= 1
        if self._readers_count == 0:
            self._writer_lock.release()
        self._reader_lock.release()

    def parse(self, text, time):
        self._begin_read()
        response = self.interpreter.parse(text, time)
        self._end_read()
        return response

    def unload(self):
        self._writer_lock.acquire()
        try:
            self.interpreter = None
            self.name = None
            self.path = None
            self.fingerprint = None
        finally:
            self._writer_lock.release()

    def is_loaded(self, model_name: Optional[Text] = None) -> bool:
        if self.interpreter is None:
            return False

        if model_name is not None:
            given_name = model_name.replace(".tar.gz", "")
            set_name = self.name.replace(".tar.gz", "")

            if given_name != set_name:
                return False

        return True

    def update_model(
        self, component_builder: ComponentBuilder, model_dir: Text, model_name: Text
    ) -> bool:
        # unload current model
        self.unload()

        self._begin_read()
        # noinspection PyUnusedLocal
        status = False

        logger.debug("Loading model from directory '{}'.".format(model_dir))

        self._loader_lock.acquire()
        try:
            self.interpreter = interpreter_for_model(component_builder, model_dir)
            self.path = model_dir
            self.name = model_name
            status = True
        finally:
            self._loader_lock.release()

        self._end_read()

        return status

    @staticmethod
    def load_local_model(dir: Text, component_builder: ComponentBuilder) -> "NLUModel":
        if os.path.isfile(dir):
            model_archive = dir
        else:
            model_archive = get_latest_model(dir)

        if model_archive is None:
            logger.warning("Could not load local model in '{}'".format(dir))
            return NLUModel.fallback_model(component_builder)

        working_directory = tempfile.mkdtemp()
        unpacked_model = model.unpack_model(model_archive, working_directory)
        _, nlu_model = model.get_model_subdirectories(unpacked_model)

        model_path = nlu_model if os.path.exists(nlu_model) else unpacked_model

        name = os.path.basename(model_archive)
        interpreter = interpreter_for_model(component_builder, model_path)

        return NLUModel(name, interpreter, model_path)

    @staticmethod
    def load_from_remote_storage(
        remote_storage: Text, component_builder: ComponentBuilder, model_name: Text
    ) -> "NLUModel":
        from rasa.nlu.persistor import get_persistor

        p = get_persistor(remote_storage)
        if p is not None:
            target_path = tempfile.mkdtemp()
            p.retrieve(model_name, target_path)
            interpreter = interpreter_for_model(component_builder, target_path)

            return NLUModel(model_name, interpreter, target_path)
        else:
            raise RuntimeError("Unable to initialize persistor")

    @staticmethod
    def fallback_model(component_builder: ComponentBuilder):
        meta = Metadata(
            {
                "pipeline": [
                    {
                        "name": "KeywordIntentClassifier",
                        "class": utils.module_path_from_object(
                            KeywordIntentClassifier()
                        ),
                    }
                ]
            },
            "",
        )
        interpreter = Interpreter.create(meta, component_builder)

        return NLUModel(FALLBACK_MODEL_NAME, interpreter)
