from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime

import glob
import json
import logging
import os
import tempfile
import io

from builtins import object
from typing import Text
from future.utils import PY3

from concurrent.futures import ProcessPoolExecutor as ProcessPool
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.logger import jsonFileLogObserver, Logger

from rasa_nlu import utils
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, InvalidModelError, Interpreter
from rasa_nlu.train import do_train_in_worker

logger = logging.getLogger(__name__)


def deferred_from_future(future):
    """Converts a concurrent.futures.Future object to a twisted.internet.defer.Deferred obejct.
    See: https://twistedmatrix.com/pipermail/twisted-python/2011-January/023296.html
    """
    d = Deferred()

    def callback(future):
        e = future.exception()
        if e:
            d.errback(e)
            return
        d.callback(future.result())

    future.add_done_callback(callback)
    return d


class DataRouter(object):
    DEFAULT_MODEL_NAME = "default"

    def __init__(self, config, component_builder):
        self._training_processes = config['max_training_processes'] if config['max_training_processes'] > 0 else 1
        self.config = config
        self.responses = DataRouter._create_query_logger(config['response_log'])
        self._trainings_queued = 0
        self.model_dir = config['path']
        self.token = config['token']
        self.emulator = self.__create_emulator()
        self.component_builder = component_builder if component_builder else ComponentBuilder(use_cache=True)
        self.model_store = self.__create_model_store()
        self.pool = ProcessPool(self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

    def shutdown(self):
        """Public wrapper over the internal __del__ function"""
        self.__del__()

    @staticmethod
    def _create_query_logger(response_log_dir):
        """Creates a logger that will persist incomming queries and their results."""

        # Ensures different log files for different processes in multi worker mode
        if response_log_dir:
            # We need to generate a unique file name, even in multiprocess environments
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file_name = "rasa_nlu_log-{}-{}.log".format(timestamp, os.getpid())
            response_logfile = os.path.join(response_log_dir, log_file_name)
            # Instantiate a standard python logger, which we are going to use to log requests
            utils.create_dir_for_file(response_logfile)
            query_logger = Logger(observer=jsonFileLogObserver(io.open(response_logfile, 'a', encoding='utf8')),
                                  namespace='query-logger')
            # Prevents queries getting logged with parent logger --> might log them to stdout
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info("Logging of requests is disabled. (No 'request_log' directory configured)")
            return None

    def _add_training_to_queue(self):
        """Adds a new training process to the list of running processes."""
        self._trainings_queued += 1

    def _remove_training_from_queue(self):
        """Decreases the ongoing trainings count by one"""
        self._trainings_queued -= 1

    def __search_for_models(self):
        models = {}
        for metadata_path in glob.glob(os.path.join(self.config.path, '*/metadata.json')):
            model_name = os.path.basename(os.path.dirname(metadata_path))
            models[model_name] = model_name
        return models

    def __interpreter_for_model(self, model_path):
        metadata = DataRouter.read_model_metadata(model_path, self.config)
        return Interpreter.load(metadata, self.config, self.component_builder)

    def __create_model_store(self):
        # Fallback for users that specified the model path as a string and hence only want a single default model.
        if type(self.config.server_model_dirs) is Text:
            model_dict = {self.DEFAULT_MODEL_NAME: self.config.server_model_dirs}
        elif self.config.server_model_dirs is None:
            model_dict = self.__search_for_models()
        else:
            model_dict = self.config.server_model_dirs

        model_store = {}

        for alias, model_path in list(model_dict.items()):
            try:
                logger.info("Loading model '{}'...".format(model_path))
                model_store[alias] = self.__interpreter_for_model(model_path)
            except Exception as e:
                logger.exception("Failed to load model '{}'. Error: {}".format(model_path, e))
        if not model_store:
            meta = Metadata({"pipeline": ["intent_classifier_keyword"]}, "")
            interpreter = Interpreter.load(meta, self.config, self.component_builder)
            model_store[self.DEFAULT_MODEL_NAME] = interpreter
        return model_store

    @staticmethod
    def default_model_metadata():
        return {
            "language": None,
        }

    @staticmethod
    def load_model_from_cloud(model_dir, config):
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(config)
            if p is not None:
                p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
            else:
                raise RuntimeError("Unable to initialize persistor")
        except Exception as e:
            logger.warning("Using default interpreter, couldn't fetch model: {}".format(e))

    @staticmethod
    def read_model_metadata(model_dir, config):
        if model_dir is None:
            data = DataRouter.default_model_metadata()
            return Metadata(data, model_dir)
        else:
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(config['path'], model_dir)

            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                DataRouter.load_model_from_cloud(model_dir, config)

            return Metadata.load(model_dir)

    def __create_emulator(self):
        """Sets which NLU webservice to emulate among those supported by Rasa"""

        mode = self.config['emulate']
        if mode is None:
            from rasa_nlu.emulators import NoEmulator
            return NoEmulator()
        elif mode.lower() == 'wit':
            from rasa_nlu.emulators.wit import WitEmulator
            return WitEmulator()
        elif mode.lower() == 'luis':
            from rasa_nlu.emulators.luis import LUISEmulator
            return LUISEmulator()
        elif mode.lower() == 'api':
            from rasa_nlu.emulators.api import ApiEmulator
            return ApiEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, data):
        alias = data.get("model") or self.DEFAULT_MODEL_NAME
        if alias not in self.model_store:
            try:
                self.model_store[alias] = self.__interpreter_for_model(model_path=alias)
            except Exception as e:
                raise InvalidModelError("No model found with alias '{}'. Error: {}".format(alias, e))

        model = self.model_store[alias]
        response = model.parse(data['text'], data.get('time', None))
        if self.responses:
            self.responses.info(user_input=response, model=alias)
        return self.format_response(response)

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.
        models = glob.glob(os.path.join(self.model_dir, '*'))
        models = [model for model in models if os.path.isfile(os.path.join(model, "metadata.json"))]
        return {
            "available_models": models,
            "trainings_queued": self._trainings_queued,
            "training_workers": self._training_processes
        }

    def start_train_process(self, data, config_values):
        if PY3:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data.json", delete=False, encoding="utf-8")
            f.write(data)
        else:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data.json", delete=False)
            f.write(data.encode("utf-8"))
        f.close()
        # TODO: fix config handling
        _config = self.config.as_dict()
        for key, val in config_values.items():
            _config[key] = val
        _config["data"] = f.name
        train_config = RasaNLUConfig(cmdline_args=_config)
        logger.info("New training queued")

        def training_callback(model_path):
            self._remove_training_from_queue()
            return os.path.basename(os.path.normpath(model_path))

        self._add_training_to_queue()

        result = self.pool.submit(do_train_in_worker, train_config)
        result = deferred_from_future(result)
        result.addCallback(training_callback)

        return result
