import glob
import logging
import multiprocessing
import os
import random
import tempfile

from flask import json

from config import RasaNLUConfig
from rasa_nlu.train import do_train


class DataRouter(object):
    def __init__(self, config, interpreter, emulator):
        self.ID = random.random()
        self.config = config
        self.interpreter = interpreter
        self.emulator = emulator
        # Ensures different log files for different processes in multi worker mode
        self.logfile = config['write'].replace(".json", "-{}.json".format(os.getpid()))
        self.responses = self.create_query_logger(self.logfile)
        self.train_procs = []
        self.model_dir = config['path']
        self.token = config['token']

    def create_query_logger(self, path):
        """Creates a logger that will persist incomming queries and their results."""

        logger = logging.getLogger('query-logger')
        logger.setLevel(logging.INFO)
        ch = logging.FileHandler(path)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.propagate = False
        logger.addHandler(ch)
        return logger

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, text):
        response = self.interpreter.parse(text)
        self.responses.info(json.dumps(response, sort_keys=True))
        return response

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.
        num_trainings = len(filter(lambda p: p.is_alive(), self.train_procs))
        models = glob.glob(os.path.join(self.model_dir, 'model*'))
        return {
            "trainings_under_this_process": num_trainings,
            "available_models": models
        }

    def start_train_process(self, data):
        logging.info("Starting model training")
        f, fname = tempfile.mkstemp(suffix="_training_data.json")
        f.write(data)
        f.flush()
        f.close()
        _config = dict(self.config.items())
        _config["data"] = fname
        train_config = RasaNLUConfig(cmdline_args=_config)
        process = multiprocessing.Process(target=do_train, args=(train_config,))
        self.train_procs.append(process)
        process.start()
        logging.info("Training process {} started".format(process))
