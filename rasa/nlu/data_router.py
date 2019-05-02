import asyncio
import datetime
import logging
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional, Text

from rasa.nlu import config, utils
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.emulators import NoEmulator
from rasa.nlu.model_loader import NLUModel, load_from_server, FALLBACK_MODEL_NAME
from rasa.nlu.test import run_evaluation
from rasa.nlu.model import InvalidModelError, UnsupportedModelError
from rasa.nlu.train import do_train_in_worker
from rasa.utils.endpoints import EndpointConfig

logger = logging.getLogger(__name__)


class MaxWorkerProcessError(Exception):
    """Raised when training or evaluation is requested and the server has
        reached the max count of worker processes.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = (
            "The server has reached its limit on process pool "
            "workers, it can't train or evaluate new models "
            "right now"
        )

    def __str__(self):
        return self.message


async def create_data_router(
    model_dir: Optional[Text] = None,
    max_worker_processes: int = 1,
    response_log: Optional[Text] = None,
    emulation_mode: Optional[Text] = None,
    remote_storage: Optional[Text] = None,
    component_builder: ComponentBuilder = None,
    model_server: EndpointConfig = None,
    wait_time_between_pulls: int = None,
) -> "DataRouter":
    router = DataRouter(
        model_dir,
        max_worker_processes,
        response_log,
        emulation_mode,
        remote_storage,
        component_builder,
        model_server,
        wait_time_between_pulls,
    )

    await router.load_model(router.model_dir)

    return router


class DataRouter(object):
    def __init__(
        self,
        model_dir: Optional[Text] = None,
        max_worker_processes: int = 1,
        response_log: Optional[Text] = None,
        emulation_mode: Optional[Text] = None,
        remote_storage: Optional[Text] = None,
        component_builder: ComponentBuilder = None,
        model_server: EndpointConfig = None,
        wait_time_between_pulls: int = None,
    ):
        self._worker_processes = max(max_worker_processes, 1)
        self._current_worker_processes = 0
        self.responses = self._create_query_logger(response_log)

        if model_dir is None:
            model_dir = tempfile.gettempdir()
        self.model_dir = os.path.abspath(model_dir)

        self.emulator = self._create_emulator(emulation_mode)
        self.remote_storage = remote_storage
        self.model_server = model_server
        self.wait_time_between_pulls = wait_time_between_pulls

        if component_builder:
            self.component_builder = component_builder
        else:
            self.component_builder = ComponentBuilder(use_cache=True)

        self.nlu_model = NLUModel.fallback_model(self.component_builder)

        # tensorflow sessions are not fork-safe,
        # and training processes have to be spawned instead of forked. See
        # https://github.com/tensorflow/tensorflow/issues/5448#issuecomment
        # -258934405
        multiprocessing.set_start_method("spawn", force=True)

    @staticmethod
    def _create_emulator(mode: Optional[Text]) -> NoEmulator:
        """Create emulator for specified mode.
        If no emulator is specified, we will use the Rasa NLU format."""

        if mode is None:
            return NoEmulator()
        elif mode.lower() == "wit":
            from rasa.nlu.emulators.wit import WitEmulator

            return WitEmulator()
        elif mode.lower() == "luis":
            from rasa.nlu.emulators.luis import LUISEmulator

            return LUISEmulator()
        elif mode.lower() == "dialogflow":
            from rasa.nlu.emulators.dialogflow import DialogflowEmulator

            return DialogflowEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    @staticmethod
    def _create_query_logger(response_log):
        """Create a logger that will persist incoming query results."""

        # Ensures different log files for different
        # processes in multi worker mode
        if response_log:
            # We need to generate a unique file name,
            # even in multiprocess environments
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file_name = "rasa_nlu_log-{}-{}.log".format(timestamp, os.getpid())
            response_logfile = os.path.join(response_log, log_file_name)
            # Instantiate a standard python logger,
            # which we are going to use to log requests
            utils.create_dir_for_file(response_logfile)
            # noinspection PyTypeChecker
            query_logger = logging.getLogger("query-logger")
            query_logger.setLevel(logging.INFO)
            ch = logging.FileHandler(response_logfile)
            ch.setFormatter(logging.Formatter("%(message)s"))
            query_logger.propagate = False
            query_logger.addHandler(ch)
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info(
                "Logging of requests is disabled. "
                "(No 'request_log' directory configured)"
            )
            return None

    async def load_model(self, model_path: Text):
        # model_path can point to a directory containing any number of tar.gz model
        # files or to one specific model file. If it is pointing to a directory, the
        # latest model in that directory is taken.

        if model_path is None:
            logger.warning("Could not load any model. Using fallback model.")
            self.nlu_model = NLUModel.fallback_model(self.component_builder)
            return

        try:
            if os.path.exists(model_path):
                self.nlu_model = NLUModel.load_local_model(
                    model_path, self.component_builder
                )

            elif self.model_server is not None:
                self.nlu_model = await load_from_server(
                    self.component_builder,
                    self.model_server,
                    self.wait_time_between_pulls,
                )

            elif self.remote_storage is not None:
                self.nlu_model = NLUModel.load_from_remote_storage(
                    self.remote_storage, self.component_builder, model_path
                )

            else:
                raise InvalidModelError(
                    "Model in '{}' could not be loaded.".format(model_path)
                )

            logger.debug("Loaded model '{}'".format(self.nlu_model.name))

        except Exception as e:
            logger.error("Could not load model due to {}.".format(e))
            raise

    def extract(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return self.emulator.normalise_request_json(data)

    def parse(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        model = data.get("model")

        nlu_model = self.nlu_model
        if not self.nlu_model.is_loaded(model):
            logger.warning(
                "Model with name '{}' is not loaded. Use default model.".format(model)
            )
            nlu_model = NLUModel.fallback_model(self.component_builder)

        response = nlu_model.parse(data["text"], data.get("time"))
        response["model"] = nlu_model.name

        if self.responses:
            self.responses.info(response)

        return self._format_response(response)

    def get_status(self) -> Dict[Text, Any]:
        # This will only count the trainings started from this
        # process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "max_worker_processes": self._worker_processes,
            "current_worker_processes": self._current_worker_processes,
            "loaded_model": self.nlu_model.name,
        }

    async def start_train_process(
        self,
        data_file: Text,
        train_config: RasaNLUModelConfig,
        model_name: Optional[Text] = None,
    ) -> Text:
        """Start a model training."""
        if self._worker_processes <= self._current_worker_processes:
            raise MaxWorkerProcessError

        loop = asyncio.get_event_loop()

        self._current_worker_processes += 1
        pool = ProcessPoolExecutor(max_workers=self._worker_processes)

        logger.debug("New training queued")

        task = loop.run_in_executor(
            pool,
            do_train_in_worker,
            train_config,
            data_file,
            self.model_dir,
            model_name,
            self.remote_storage,
        )

        try:
            return await task
        finally:
            self._current_worker_processes -= 1
            pool.shutdown()

    def unload_model(self, model: Text):
        """Unload a model from server memory."""
        if not self.nlu_model.is_loaded(model):
            raise InvalidModelError("Model with name '{}' is not loaded.".format(model))

        self.nlu_model.unload()

    # noinspection PyProtectedMember
    async def evaluate(
        self, data_file: Text, model: Optional[Text] = None
    ) -> Dict[Text, Any]:
        """Perform a model evaluation."""
        if not self.nlu_model.is_loaded(model):
            raise InvalidModelError("Model with name '{}' is not loaded.".format(model))

        logger.debug("Evaluation request received for model '{}'.".format(model))

        if self._worker_processes <= self._current_worker_processes:
            raise MaxWorkerProcessError

        if self.nlu_model.name == FALLBACK_MODEL_NAME:
            raise UnsupportedModelError("No model is loaded. Cannot evaluate.")

        loop = asyncio.get_event_loop()

        self._current_worker_processes += 1
        pool = ProcessPoolExecutor(max_workers=self._worker_processes)

        task = loop.run_in_executor(
            pool, run_evaluation, data_file, self.nlu_model.path
        )

        try:
            return await task
        finally:
            self._current_worker_processes -= 1
            pool.shutdown()

    def _format_response(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return self.emulator.normalise_response_json(data)
