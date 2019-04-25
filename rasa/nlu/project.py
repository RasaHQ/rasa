import time

import datetime
import logging
import os
import tempfile
import zipfile
from io import BytesIO as IOReader
from requests.exceptions import InvalidURL, RequestException
from threading import Lock, Thread
from typing import List, Optional, Text

from rasa.nlu import utils
from rasa.nlu.classifiers.keyword_intent_classifier import KeywordIntentClassifier
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.model import Interpreter, Metadata
from rasa.nlu.utils import is_url
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)

MODEL_NAME_PREFIX = "model_"

FALLBACK_MODEL_NAME = "fallback"

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

STATUS_READY = 0
STATUS_TRAINING = 1
STATUS_FAILED = -1


async def load_from_server(
    component_builder: Optional[ComponentBuilder] = None,
    project: Optional[Text] = None,
    project_dir: Optional[Text] = None,
    remote_storage: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    wait_time_between_pulls: Optional[int] = None,
) -> "Project":
    """Load a persisted model from a server."""

    project = Project(
        component_builder=component_builder,
        project=project,
        project_dir=project_dir,
        remote_storage=remote_storage,
        pull_models=True,
    )

    await _update_model_from_server(model_server, project)

    if wait_time_between_pulls:
        # continuously pull the model every `wait_time_between_pulls` seconds
        start_model_pulling_in_worker(model_server, wait_time_between_pulls, project)
    return project


async def _update_model_from_server(
    model_server: EndpointConfig, project: "Project"
) -> None:
    """Load a zipped Rasa NLU model from a URL and update the passed

    project."""
    if not is_url(model_server.url):
        raise InvalidURL(model_server)

    model_directory = tempfile.mkdtemp()

    new_model_fingerprint, filename = await _pull_model_and_fingerprint(
        model_server, model_directory, project.fingerprint
    )
    if new_model_fingerprint:
        model_name = _get_remote_model_name(filename)
        project.fingerprint = new_model_fingerprint
        project.update_model_from_dir_and_unload_others(model_directory, model_name)
    else:
        logger.debug("No new model found at URL {}".format(model_server.url))


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
            "Model server didn't find a model for our request. "
            "Probably no one did train a model for the project "
            "and tag combination yet."
        )
        return None, None
    elif response.status_code != 200:
        logger.warning(
            "Tried to fetch model from server, but server response "
            "status code is {}. We'll retry later..."
            "".format(response.status_code)
        )
        return None, None

    zip_ref = zipfile.ZipFile(IOReader(response.content))
    zip_ref.extractall(model_directory)
    logger.debug("Unzipped model to {}".format(os.path.abspath(model_directory)))

    # get the new fingerprint and filename
    return response.headers.get("ETag"), response.headers.get("filename")


async def _run_model_pulling_worker(
    model_server: EndpointConfig, wait_time_between_pulls: int, project: "Project"
) -> None:
    while True:
        await _update_model_from_server(model_server, project)
        time.sleep(wait_time_between_pulls)


def start_model_pulling_in_worker(
    model_server: Optional[EndpointConfig],
    wait_time_between_pulls: int,
    project: "Project",
) -> None:
    worker = Thread(
        target=_run_model_pulling_worker,
        args=(model_server, wait_time_between_pulls, project),
    )
    worker.setDaemon(True)
    worker.start()


class Project(object):
    def __init__(
        self,
        component_builder=None,
        project=None,
        project_dir=None,
        remote_storage=None,
        fingerprint=None,
        pull_models=None,
    ):
        self._component_builder = component_builder
        self._models = {}
        self.status = STATUS_READY
        self.current_worker_processes = 0
        self._reader_lock = Lock()
        self._loader_lock = Lock()
        self._writer_lock = Lock()
        self._readers_count = 0
        self._path = None
        self._project = project
        self.remote_storage = remote_storage
        self.fingerprint = fingerprint
        self.pull_models = pull_models
        self.error_message = None

        if project and project_dir:
            self._path = os.path.join(project_dir, project)
        self._search_for_models()

    def _begin_read(self):
        # Readers-writer lock basic double mutex implementation
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

    def _load_local_model(self, requested_model_name=None):
        if requested_model_name is None:  # user want latest model
            # NOTE: for better parse performance, currently although
            # user may want latest model by set requested_model_name
            # explicitly to None, we are not refresh model list
            # from local and cloud which is pretty slow.
            # User can specific requested_model_name to the latest model name,
            # then model will be cached, this is a kind of workaround to
            # refresh latest project model.
            # BTW if refresh function is wanted, maybe add implement code to
            # `_latest_project_model()` is a good choice.

            logger.debug("No model specified. Using default")
            return self._latest_project_model()

        elif requested_model_name in self._models:  # model exists in cache
            return requested_model_name

        return None  # local model loading failed!

    def _dynamic_load_model(self, requested_model_name: Text = None):

        # If the Project was configured to pull models from a
        # server, only one model is in memory at a time.
        # Use this model if it exists.
        if self.pull_models and requested_model_name is None:
            for model, interpreter in self._models.items():
                if interpreter is not None:
                    return model

        # first try load from local cache
        local_model = self._load_local_model(requested_model_name)
        if local_model:
            return local_model

        # now model not exists in model list cache
        # refresh model list from local and cloud

        # NOTE: if a malicious user sent lots of requests
        # with not existing model will cause performance issue.
        # because get anything from cloud is a time-consuming task
        self._search_for_models()

        # retry after re-fresh model cache
        local_model = self._load_local_model(requested_model_name)
        if local_model:
            return local_model

        # still not found user specified model
        logger.warning("Invalid model requested. Using default")
        return self._latest_project_model()

    def parse(self, text, parsing_time=None, requested_model_name=None):
        self._begin_read()

        model_name = self._dynamic_load_model(requested_model_name)

        self._loader_lock.acquire()
        try:
            if not self._models.get(model_name):
                interpreter = self._interpreter_for_model(model_name)
                self._models[model_name] = interpreter
        finally:
            self._loader_lock.release()

        response = self._models[model_name].parse(text, parsing_time)
        response["project"] = self._project
        response["model"] = model_name

        self._end_read()

        return response

    def load_model(self):
        self._begin_read()
        status = False
        model_name = self._dynamic_load_model()
        logger.debug("Loading model %s", model_name)

        self._loader_lock.acquire()
        try:
            if not self._models.get(model_name):
                interpreter = self._interpreter_for_model(model_name)
                self._models[model_name] = interpreter
                status = True
        finally:
            self._loader_lock.release()

        self._end_read()

        return status

    def update_model_from_dir_and_unload_others(
        self, model_dir: Text, model_name: Text
    ) -> bool:
        # unload all loaded models
        for model in self._list_loaded_models():
            self.unload(model)

        self._begin_read()
        # noinspection PyUnusedLocal
        status = False

        logger.debug(
            "Loading model '{}' from directory '{}'.".format(model_name, model_dir)
        )

        self._loader_lock.acquire()
        try:
            interpreter = self._interpreter_for_model(model_name, model_dir)
            self._models[model_name] = interpreter
            status = True
        finally:
            self._loader_lock.release()

        self._end_read()

        return status

    def update(self, model_name):
        self._writer_lock.acquire()
        self._models[model_name] = None
        self._writer_lock.release()

    def unload(self, model_name):
        self._writer_lock.acquire()
        try:
            del self._models[model_name]
            self._models[model_name] = None
            return model_name
        finally:
            self._writer_lock.release()

    def _latest_project_model(self):
        """Retrieves the latest trained model for an project"""

        models = {
            model[len(MODEL_NAME_PREFIX) :]: model
            for model in self._models.keys()
            if model.startswith(MODEL_NAME_PREFIX)
        }
        if models:
            time_list = [
                datetime.datetime.strptime(parse_time, "%Y%m%d-%H%M%S")
                for parse_time, model in models.items()
            ]
            return models[max(time_list).strftime("%Y%m%d-%H%M%S")]
        else:
            return FALLBACK_MODEL_NAME

    def _fallback_model(self):
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
        return Interpreter.create(meta, self._component_builder)

    def _search_for_models(self):
        model_names = (
            self._list_models_in_dir(self._path) + self._list_models_in_cloud()
        )
        if not model_names:
            if FALLBACK_MODEL_NAME not in self._models:
                self._models[FALLBACK_MODEL_NAME] = self._fallback_model()
        else:
            for model in set(model_names):
                if model not in self._models:
                    self._models[model] = None

    def _interpreter_for_model(self, model_name, model_dir=None):
        metadata = self._read_model_metadata(model_name, model_dir)
        return Interpreter.create(metadata, self._component_builder)

    def _read_model_metadata(self, model_name, model_dir):
        if model_name is None:
            data = Project._default_model_metadata()
            return Metadata(data, model_name)
        else:
            if model_dir is not None:
                path = model_dir
            elif not os.path.isabs(model_name) and self._path:
                path = os.path.join(self._path, model_name)
            else:
                path = model_name

            # download model from cloud storage if needed and possible
            if not os.path.isdir(path):
                self._load_model_from_cloud(model_name, path)

            return Metadata.load(path)

    def as_dict(self):
        status = "ready"
        error_message = None
        if self.status == STATUS_TRAINING:
            status = "training"
        elif self.status == STATUS_FAILED:
            status = "failed"
            error_message = self.error_message

        return {
            "status": status,
            "error_message": error_message,
            "current_worker_processes": self.current_worker_processes,
            "available_models": list(self._models.keys()),
            "loaded_models": self._list_loaded_models(),
        }

    def _list_loaded_models(self):
        models = []
        for model, interpreter in self._models.items():
            if interpreter is not None:
                models.append(model)
        return models

    def _list_models_in_cloud(self) -> List[Text]:

        try:
            from rasa.nlu.persistor import get_persistor

            p = get_persistor(self.remote_storage)
            if p is not None:
                return p.list_models(self._project)
            else:
                return []
        except Exception as e:
            logger.warning(
                "Failed to list models of project {}. {}".format(self._project, e)
            )
            return []

    def _load_model_from_cloud(self, model_name, target_path):
        try:
            from rasa.nlu.persistor import get_persistor

            p = get_persistor(self.remote_storage)
            if p is not None:
                p.retrieve(model_name, self._project, target_path)
            else:
                raise RuntimeError("Unable to initialize persistor")
        except Exception as e:
            logger.warning(
                "Using default interpreter, couldn't fetch model: {}".format(e)
            )
            raise  # re-raise this exception because nothing we can do now

    @staticmethod
    def _default_model_metadata():
        return {"language": None}

    @staticmethod
    def _list_models_in_dir(path):
        if not path or not os.path.isdir(path):
            return []
        else:
            return [
                os.path.relpath(model, path)
                for model in utils.list_subdirectories(path)
            ]
