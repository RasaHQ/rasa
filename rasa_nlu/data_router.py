from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object
import datetime
import glob
import json
import logging
import multiprocessing
import os
import tempfile

from flask import json
from typing import Text

import rasa_nlu.components
from rasa_nlu import registry
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, InvalidModelError, Interpreter
from rasa_nlu.train import do_train
from rasa_nlu import utils


class InterpreterBuilder(object):
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        # Reuse nlp and featurizers where possible to save memory
        self.component_cache = {}

    def __get_component(self, component_name, meta, context, model_config):
        component_class = registry.get_component_class(component_name)
        cache_key = component_class.cache_key(meta)
        if cache_key is not None and self.use_cache and cache_key in self.component_cache:
            component = self.component_cache[cache_key]
        else:
            component = registry.load_component_by_name(component_name, context, model_config)
            if cache_key is not None and self.use_cache:
                self.component_cache[cache_key] = component
        return component

    def create_interpreter(self, meta, config):
        context = {"model_dir": meta.model_dir}

        model_config = dict(list(config.items()))
        model_config.update(meta.metadata)

        pipeline = []

        for component_name in meta.pipeline:
            try:
                component = self.__get_component(component_name, meta, context, model_config)
                rasa_nlu.components.init_component(component, context, model_config)
                pipeline.append(component)
            except rasa_nlu.components.MissingArgumentError as e:
                raise Exception("Failed to initialize component '{}'. {}".format(component_name, e.message))

        return Interpreter(pipeline, context, model_config)


class DataRouter(object):
    DEFAULT_MODEL_NAME = "default"

    def __init__(self, config):
        self.config = config
        self.responses = DataRouter._create_query_logger(config['response_log'])
        self.train_procs = []
        self.model_dir = config['path']
        self.token = config['token']
        self.emulator = self.__create_emulator()
        self.interpreter_builder = InterpreterBuilder()
        self.model_store = self.__create_model_store()

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
            logger = logging.getLogger('query-logger')
            logger.setLevel(logging.INFO)
            utils.create_dir_for_file(response_logfile)
            ch = logging.FileHandler(response_logfile)
            ch.setFormatter(logging.Formatter('%(message)s'))
            logger.propagate = False  # Prevents queries getting logged with parent logger --> might log them to stdout
            logger.addHandler(ch)
            logging.info("Logging requests to '{}'.".format(response_logfile))
            return logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logging.info("Logging of requests is disabled. (No 'request_log' directory configured)")
            return None

    def __search_for_models(self):
        models = {}
        for metadata_path in glob.glob(os.path.join(self.config.path, '*/metadata.json')):
            model_name = os.path.basename(os.path.dirname(metadata_path))
            models[model_name] = model_name
        return models

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
                logging.info("Loading model '{}'...".format(model_path))
                metadata = DataRouter.read_model_metadata(model_path, self.config)
                interpreter = self.interpreter_builder.create_interpreter(metadata, self.config)
                model_store[alias] = interpreter
            except Exception as e:
                logging.error("Failed to load model '{}'. Error: {}".format(model_path, e))
        if not model_store:
            meta = Metadata({"pipeline": ["intent_classifier_keyword"]}, "")
            interpreter = self.interpreter_builder.create_interpreter(meta, self.config)
            model_store[self.DEFAULT_MODEL_NAME] = interpreter
        return model_store

    @staticmethod
    def default_model_metadata():
        return {
            "language": None,
        }

    @staticmethod
    def load_model_from_s3(model_dir, config):
        try:
            from rasa_nlu.persistor import Persistor
            p = Persistor(config['path'], config['aws_region'], config['bucket_name'])
            p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
        except Exception as e:
            logging.warn("Using default interpreter, couldn't fetch model: {}".format(e.message))

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
                DataRouter.load_model_from_s3(model_dir, config)

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
        alias = data.get("model") or self.DEFAULT_MODEL_NAME
        if alias not in self.model_store:
            raise InvalidModelError("No model found with alias '{}'".format(alias))
        else:
            model = self.model_store[alias]
            response = model.parse(data['text'])
            if self.responses:
                self.responses.info(json.dumps(response, sort_keys=True))
            return self.format_response(response)

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.
        num_trainings = len([p for p in self.train_procs if p.is_alive()])
        models = glob.glob(os.path.join(self.model_dir, 'model*'))
        return {
            "trainings_under_this_process": num_trainings,
            "available_models": models
        }

    def start_train_process(self, data):
        logging.info("Starting model training")
        f = tempfile.NamedTemporaryFile("w+", suffix="_training_data.json", delete=False)
        f.write(data)
        f.close()
        _config = dict(list(self.config.items()))
        _config["data"] = f.name
        train_config = RasaNLUConfig(cmdline_args=_config)
        process = multiprocessing.Process(target=do_train, args=(train_config,))
        self.train_procs.append(process)
        process.start()
        logging.info("Training process {} started".format(process))
