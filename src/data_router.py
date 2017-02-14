import glob
import logging
import multiprocessing
import os
import random
from flask import json

from config import RasaNLUConfig
from rasa_nlu.train import do_train


class DataRouter(object):
    def __init__(self, config, interpreter, emulator):
        self.ID = random.random()
        self.config = config
        self.interpreter = interpreter
        self.emulator = emulator
        self.logfile = config['write'].replace(".json", "-{}.json".format(os.getpid()))
        self.responses = self.create_query_logger(self.logfile)
        self.train_proc = None
        self.model_dir = config['path']
        self.token = config['token']

    def create_query_logger(self, path):
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
        result = self.interpreter.parse(text)
        self.responses.info(json.dumps(result, sort_keys=True))
        return result

    def format(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        if self.train_proc is not None:
            training = self.train_proc.is_alive()
        else:
            training = False

        models = glob.glob(os.path.join(self.model_dir, 'model*'))
        return {
            "training": training,
            "available_models": models
        }

    def start_train_proc(self, data):
        logging.info("starting train")
        if self.train_proc is not None and self.train_proc.is_alive():
            self.train_proc.terminate()
            logging.info("training process {0} killed".format(self.train_proc))

        fname = 'tmp_training_data.json'
        with open(fname, 'w') as f:
            f.write(data)
        _config = dict(self.config.items())
        _config["data"] = fname
        train_config = RasaNLUConfig(cmdline_args=_config)

        self.train_proc = multiprocessing.Process(target=do_train, args=(train_config,))
        self.train_proc.start()
        logging.info("training process {0} started".format(self.train_proc))
