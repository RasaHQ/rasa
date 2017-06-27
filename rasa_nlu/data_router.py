from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import logging
import os
import tempfile
import io
import json

from builtins import object
from typing import Text

from rasa_nlu import utils
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, InvalidModelError, Interpreter
from rasa_nlu.train import do_train

logger = logging.getLogger(__name__)


class DataRouter(object):
    DEFAULT_AGENT_NAME = "default"

    def __init__(self, config, component_builder):
        self.config = config
        self.responses = DataRouter._create_query_logger(config['response_log'])
        self._train_procs = []
        self.model_dir = config['path']
        self.token = config['token']
        self.emulator = self.__create_emulator()
        self.component_builder = component_builder if component_builder else ComponentBuilder(use_cache=True)
        self.agent_store = self.__create_agent_store()

    @staticmethod
    def _latest_agent_model(agent_path):
        """Retrieves the latest trained model for an agent"""
        agent_models = {model[6:]: model for model in os.listdir(agent_path)}
        time_list = [datetime.datetime.strptime(time, '%Y%m%d-%H%M%S') for time, model in agent_models.items()]

        return agent_models[max(time_list).strftime('%Y%m%d-%H%M%S')]

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
            query_logger = logging.getLogger('query-logger')
            query_logger.setLevel(logging.INFO)
            utils.create_dir_for_file(response_logfile)
            ch = logging.FileHandler(response_logfile)
            ch.setFormatter(logging.Formatter('%(message)s'))
            # Prevents queries getting logged with parent logger --> might log them to stdout
            query_logger.propagate = False
            query_logger.addHandler(ch)
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info("Logging of requests is disabled. (No 'request_log' directory configured)")
            return None

    def _remove_finished_procs(self):
        """Remove finished training processes from the list of running training processes."""
        self._train_procs = [p for p in self._train_procs if p.is_alive()]

    def _add_train_proc(self, p):
        """Adds a new training process to the list of running processes."""
        self._train_procs.append(p)

    @property
    def train_procs(self):
        """Instead of accessing the `_train_procs` property directly, this method will ensure that trainings that
        are finished will be removed from the list."""

        self._remove_finished_procs()
        return self._train_procs

    def train_proc_ids(self):
        """Returns the ids of the running trainings processes."""
        return [p.ident for p in self.train_procs]

    def __search_for_models(self):
        models = {}
        for agent_dirname in os.listdir(self.config.path):
            agent_path = os.path.join(self.config['path'], agent_dirname)
            models[agent_dirname] = DataRouter._latest_agent_model(agent_path)

        return models

    def __interpreter_for_model(self, latest_model_path):
        metadata = DataRouter.read_model_metadata(latest_model_path, self.config)
        return Interpreter.load(metadata, self.config, self.component_builder)

    def __create_agent_store(self):
        # Fallback for users that specified the model path as a string and hence only want a single default model.
        if type(self.config.server_model_dirs) is Text:
            model_dict = {self.DEFAULT_AGENT_NAME: self.config.server_model_dirs}
        elif self.config.server_model_dirs is None:
            model_dict = self.__search_for_models()
        else:
            model_dict = self.config.server_model_dirs

        agent_store = {}

        for agent, model_dirname in list(model_dict.items()):
            try:
                model_path = os.path.join(self.config['path'], agent, model_dirname)
                logger.info("Loading model '{}' for agent '{}'...".format(model_dirname, agent))
                agent_store[agent] = self.__interpreter_for_model(model_path)
            except Exception as e:
                logger.exception("Failed to load model '{}' for agent '{}'. Error: {}".format(model_dirname, agent, e))
        if not agent_store:
            meta = Metadata({"pipeline": ["intent_classifier_keyword"]}, "")
            interpreter = Interpreter.load(meta, self.config, self.component_builder)
            agent_store[self.DEFAULT_AGENT_NAME] = interpreter
        return agent_store

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
            logger.warn("Using default interpreter, couldn't fetch model: {}".format(e))

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
        agent = data.get("model") or self.DEFAULT_AGENT_NAME
        if agent not in self.agent_store:
            try:
                self.agent_store[agent] = self.__interpreter_for_model(latest_model_path=agent)
            except Exception as e:
                raise InvalidModelError("No agent found with name '{}'. Error: {}".format(agent, e))

        model = self.agent_store[agent]
        response = model.parse(data['text'], data.get('time', None))
        if self.responses:
            log = {"user_input": response, "model": agent, "time": datetime.datetime.now().isoformat()}
            self.responses.info(json.dumps(log, sort_keys=True))
        return self.format_response(response)

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.
        num_trainings = len(self.train_procs)
        models = glob.glob(os.path.join(self.model_dir, '*'))
        models = [model for model in models if os.path.isfile(os.path.join(model, "metadata.json"))]
        return {
            "trainings_under_this_process": num_trainings,
            "available_models": models,
            "training_process_ids": self.train_proc_ids()
        }

    def start_train_process(self, data, config_values):
        logger.info("Starting model training")
        f = tempfile.NamedTemporaryFile("w+", suffix="_training_data.json", delete=False)
        f.write(data)
        f.close()
        # TODO: fix config handling
        _config = self.config.as_dict()
        for key, val in config_values.items():
            _config[key] = val
        _config["data"] = f.name
        train_config = RasaNLUConfig(cmdline_args=_config)
        process = multiprocessing.Process(target=do_train, args=(train_config, self.component_builder))
        self._add_train_proc(process)
        process.start()
        logger.info("Training process {} started".format(process))
