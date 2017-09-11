from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import json
import logging
import os
import tempfile
import io

import twisted
from builtins import object
from typing import Text, Dict, Any
from future.utils import PY3

from concurrent.futures import ProcessPoolExecutor as ProcessPool
from twisted.internet.defer import Deferred
from twisted.logger import jsonFileLogObserver, Logger

from rasa_nlu import utils
from rasa_nlu.agent import Agent
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, InvalidModelError, Interpreter
from rasa_nlu.train import do_train_in_worker

logger = logging.getLogger(__name__)


class AlreadyTrainingError(Exception):
    """Raised when a training request is received for an Agent already being trained.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = 'The agent is already being trained!'

    def __str__(self):
        return self.message


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
    DEFAULT_AGENT_NAME = "default"

    def __init__(self, config, component_builder):
        self._training_processes = config['max_training_processes'] if config['max_training_processes'] > 0 else 1
        self.config = config
        self.responses = DataRouter._create_query_logger(config['response_log'])
        self.model_dir = config['path']
        self.token = config['token']
        self.emulator = self._create_emulator()
        self.component_builder = component_builder if component_builder else ComponentBuilder(use_cache=True)
        self.agent_store = self._create_agent_store()
        self.pool = ProcessPool(self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

    @staticmethod
    def _create_query_logger(response_log_dir):
        """Creates a logger that will persist incoming queries and their results."""

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

    def _create_agent_store(self):
        agents = []

        if os.path.isdir(self.config['path']):
            agents = os.listdir(self.config['path'])

        agent_store = {}

        for agent in agents:
            agent_store[agent] = Agent(self.config, self.component_builder, agent)

        if not agent_store:
            agent_store[self.DEFAULT_AGENT_NAME] = Agent()
        return agent_store

    def _create_emulator(self):
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
        agent = data.get("agent") or self.DEFAULT_AGENT_NAME
        model = data.get("model")

        if agent not in self.agent_store:
            agents = os.listdir(self.config['path'])
            if agent not in agents:
                raise InvalidModelError("No agent found with name '{}'.".format(agent))
            else:
                try:
                    self.agent_store[agent] = Agent(self.config, self.component_builder, agent)
                except Exception as e:
                    raise InvalidModelError("No agent found with name '{}'. Error: {}".format(agent, e))

        response, used_model = self.agent_store[agent].parse(data['text'], data.get('time', None), model)

        if self.responses:
            self.responses.info(user_input=response, agent=agent, model=used_model)
        return self.format_response(response)

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "available_agents": {name: agent.as_dict() for name, agent in self.agent_store.items()}
        }

    def start_train_process(self, data, config_values):
        # type: (Text, Dict[Text, Any]) -> Deferred
        """Start a model training."""

        if PY3:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data", delete=False, encoding="utf-8")
            f.write(data)
        else:
            f = tempfile.NamedTemporaryFile("w+", suffix="_training_data", delete=False)
            f.write(data.encode("utf-8"))
        f.close()
        # TODO: fix config handling
        _config = self.config.as_dict()
        for key, val in config_values.items():
            _config[key] = val
        _config["data"] = f.name
        train_config = RasaNLUConfig(cmdline_args=_config)

        agent = _config.get("name")
        if not agent:
            raise InvalidModelError("Missing agent name to train")
        elif agent in self.agent_store:
            if self.agent_store[agent].status == 1:
                raise AlreadyTrainingError
            else:
                self.agent_store[agent].status = 1
        elif agent not in self.agent_store:
            self.agent_store[agent] = Agent(self.config, self.component_builder, agent)
            self.agent_store[agent].status = 1

        def training_callback(model_path):
            model_dir = os.path.basename(os.path.normpath(model_path))
            self.agent_store[agent].update(model_dir)
            return model_dir

        logger.info("New training queued")

        result = self.pool.submit(do_train_in_worker, train_config)
        result = deferred_from_future(result)
        result.addCallback(training_callback)

        return result
