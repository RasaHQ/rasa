import asyncio
import datetime
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional, Text

from rasa.nlu import config, utils
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.emulators import NoEmulator
from rasa.nlu.model_loader import NLUModel, load_from_server
from rasa.nlu.test import run_evaluation
from rasa.nlu.model import InvalidProjectError
from rasa.nlu.train import do_train_in_worker


logger = logging.getLogger(__name__)


class MaxTrainingError(Exception):
    """Raised when a training is requested and the server has
        reached the max count of training processes.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = "The server can't train more models right now!"

    def __str__(self):
        return self.message


class DataRouter(object):
    def __init__(
        self,
        model_dir=None,
        max_training_processes=1,
        response_log=None,
        emulation_mode=None,
        remote_storage=None,
        component_builder=None,
        model_server=None,
        wait_time_between_pulls=None,
    ):

        self._training_processes = max(max_training_processes, 1)
        self._current_training_processes = 0
        self.responses = self._create_query_logger(response_log)

        self.model_dir = config.make_path_absolute(model_dir)

        self.emulator = self._create_emulator(emulation_mode)
        self.remote_storage = remote_storage
        self.model_server = model_server
        self.wait_time_between_pulls = wait_time_between_pulls

        if component_builder:
            self.component_builder = component_builder
        else:
            self.component_builder = ComponentBuilder(use_cache=True)

        self.nlu_model: NLUModel = NLUModel.fallback_model(self.component_builder)

        # TODO: Should be moved to separate method
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
        loop.run_until_complete(self._load_model(self.model_dir))
        loop.close()

        # tensorflow sessions are not fork-safe,
        # and training processes have to be spawned instead of forked. See
        # https://github.com/tensorflow/tensorflow/issues/5448#issuecomment
        # -258934405
        multiprocessing.set_start_method("spawn", force=True)

        self.pool = ProcessPoolExecutor(max_workers=self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

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

    async def _load_model(self, model_dir: Text):
        if model_dir is None:
            logger.info("Could not load any model. Use fallback model.")
            self.nlu_model = NLUModel.fallback_model(self.component_builder)
            return

        if os.path.exists(model_dir):
            self.nlu_model = NLUModel.load_local_model(
                model_dir, self.component_builder
            )

        elif self.model_server is not None:
            self.nlu_model = await load_from_server(
                self.component_builder, self.model_server, self.wait_time_between_pulls
            )

        elif self.remote_storage is not None:
            self.nlu_model = NLUModel.load_from_remote_storage(
                self.remote_storage, self.component_builder, model_dir
            )

    def extract(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return self.emulator.normalise_request_json(data)

    def parse(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        model = data.get("model")

        if not self.nlu_model.is_loaded(model):
            raise InvalidProjectError(
                "Model with name '{}' is not loaded.".format(model)
            )

        response = self.nlu_model.parse(data["text"], data.get("time"))
        response["model"] = self.nlu_model.name

        if self.responses:
            self.responses.info(response)

        return self._format_response(response)

    def get_status(self) -> Dict[Text, Any]:
        # This will only count the trainings started from this
        # process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "max_training_processes": self._training_processes,
            "current_training_processes": self._current_training_processes,
            "loaded_model": self.nlu_model.name,
        }

    async def start_train_process(
        self,
        data_file: Text,
        train_config: RasaNLUModelConfig,
        model_name: Optional[Text] = None,
    ):
        """Start a model training."""
        if self._training_processes <= self._current_training_processes:
            raise MaxTrainingError

        loop = asyncio.get_event_loop()

        logger.debug("New training queued")

        self._current_training_processes += 1

        task = loop.run_in_executor(
            self.pool,
            do_train_in_worker,
            train_config,
            data_file,
            self.model_dir,
            model_name,
            self.remote_storage,
        )

        try:
            return await task
        except Exception as e:
            logger.warning(e)
        finally:
            self._current_training_processes -= 1

    # noinspection PyProtectedMember
    def evaluate(self, data: Text, model: Optional[Text] = None) -> Dict[Text, Any]:
        """Perform a model evaluation."""
        if not self.nlu_model.is_loaded(model):
            raise InvalidProjectError(
                "Model with name '{}' is not loaded.".format(model)
            )

        file_name = utils.create_temporary_file(data, "_training_data")

        return run_evaluation(
            data_path=file_name, model=self.nlu_model.interpreter, errors_filename=None
        )

    def unload_model(self, model: Text):
        """Unload a model from server memory."""
        if not self.nlu_model.is_loaded(model):
            raise InvalidProjectError(
                "Model with name '{}' is not loaded.".format(model)
            )

        self.nlu_model.unload()

    def load_model(self):
        # TODO
        pass

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

    def _format_response(self, data: Dict[Text, Any]) -> Dict[Text, Any]:
        return self.emulator.normalise_response_json(data)
