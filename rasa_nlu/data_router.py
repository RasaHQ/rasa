from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import glob
import io
import logging
import os
import tempfile

from builtins import object
from typing import Text, Dict, Any
from future.utils import PY3

from concurrent.futures import ProcessPoolExecutor as ProcessPool
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.logger import jsonFileLogObserver, Logger

from rasa_nlu import utils
from rasa_nlu.project import Project
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, InvalidProjectError, Interpreter
from rasa_nlu.train import do_train_in_worker

logger = logging.getLogger(__name__)

# in some execution environments `reactor.callFromThread` can not be called as it will result in a deadlock as
# the `callFromThread` queues the function to be called by the reactor which only happens after the call to `yield`.
# Unfortunately, the test is blocked there because `app.flush()` needs to be called to allow the fake server to
# respond and change the status of the Deferred on which the client is yielding. Solution: during tests we will set
# this Flag to `False` to directly run the calls instead of wrapping them in `callFromThread`.
DEFERRED_RUN_IN_REACTOR_THREAD = True


class AlreadyTrainingError(Exception):
    """Raised when a training request is received for an Project already being trained.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = 'The project is already being trained!'

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
            if DEFERRED_RUN_IN_REACTOR_THREAD:
                reactor.callFromThread(d.errback, e)
            else:
                d.errback(e)
        else:
            if DEFERRED_RUN_IN_REACTOR_THREAD:
                reactor.callFromThread(d.callback, future.result())
            else:
                d.callback(future.result())

    future.add_done_callback(callback)
    return d


class DataRouter(object):
    def __init__(self, config, component_builder):
        self._training_processes = config['max_training_processes'] if config['max_training_processes'] > 0 else 1
        self.config = config
        self.responses = self._create_query_logger(config)
        self.model_dir = config['path']
        self.emulator = self._create_emulator()
        self.component_builder = component_builder if component_builder else ComponentBuilder(use_cache=True)
        self.project_store = self._create_project_store()
        self.pool = ProcessPool(self._training_processes)

    def __del__(self):
        """Terminates workers pool processes"""
        self.pool.shutdown()

    def _create_query_logger(self, config):
        """Creates a logger that will persist incoming queries and their results."""

        response_log_dir = config['response_log']
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

    def _create_project_store(self):
        projects = []

        if os.path.isdir(self.config['path']):
            projects = os.listdir(self.config['path'])

        project_store = {}

        for project in projects:
            project_store[project] = Project(self.config, self.component_builder, project)

        if not project_store:
            project_store[RasaNLUConfig.DEFAULT_PROJECT_NAME] = Project(self.config)
        return project_store

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
        elif mode.lower() == 'dialogflow':
            from rasa_nlu.emulators.dialogflow import DialogflowEmulator
            return DialogflowEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, data):
        project = data.get("project") or RasaNLUConfig.DEFAULT_PROJECT_NAME
        model = data.get("model")

        if project not in self.project_store:
            projects = self._list_projects(self.config['path'])
            if project not in projects:
                raise InvalidProjectError("No project found with name '{}'.".format(project))
            else:
                try:
                    self.project_store[project] = Project(self.config, self.component_builder, project)
                except Exception as e:
                    raise InvalidProjectError("Unable to load project '{}'. Error: {}".format(project, e))

        response, used_model = self.project_store[project].parse(data['text'], data.get('time', None), model)

        if self.responses:
            self.responses.info('', user_input=response, project=project, model=used_model)
        return self.format_response(response)

    @staticmethod
    def _list_projects(path):
        """List the projects in the path, ignoring hidden directories."""
        return [fn
                for fn in glob.glob(os.path.join(path, '*'))
                if os.path.isdir(fn)]

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "available_projects": {name: project.as_dict() for name, project in self.project_store.items()}
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

        project = _config.get("project")
        if not project:
            raise InvalidProjectError("Missing project name to train")
        elif project in self.project_store:
            if self.project_store[project].status == 1:
                raise AlreadyTrainingError
            else:
                self.project_store[project].status = 1
        elif project not in self.project_store:
            self.project_store[project] = Project(self.config, self.component_builder, project)
            self.project_store[project].status = 1

        def training_callback(model_path):
            model_dir = os.path.basename(os.path.normpath(model_path))
            self.project_store[project].update(model_dir)
            return model_dir

        def training_errback(failure):
            target_project = self.project_store.get(failure.value.failed_target_project)
            if target_project:
                target_project.status = 0
            return failure

        logger.debug("New training queued")

        result = self.pool.submit(do_train_in_worker, train_config)
        result = deferred_from_future(result)
        result.addCallback(training_callback)
        result.addErrback(training_errback)

        return result
