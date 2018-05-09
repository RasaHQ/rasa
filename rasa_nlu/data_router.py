from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob
import io
import logging
import tempfile

import datetime
import os
from builtins import object
from concurrent.futures import ProcessPoolExecutor as ProcessPool
from future.utils import PY3
from rasa_nlu.training_data import Message

from rasa_nlu import utils, config
from rasa_nlu.components import ComponentBuilder
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.evaluate import get_evaluation_metrics, clean_intent_labels
from rasa_nlu.model import InvalidProjectError
from rasa_nlu.project import Project
from rasa_nlu.train import do_train_in_worker
from rasa_nlu.training_data.loading import load_data
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.logger import jsonFileLogObserver, Logger
from typing import Text, Dict, Any, Optional, List
from rasa_nlu.project_manager import ProjectManager

logger = logging.getLogger(__name__)

# in some execution environments `reactor.callFromThread`
# can not be called as it will result in a deadlock as
# the `callFromThread` queues the function to be called
# by the reactor which only happens after the call to `yield`.
# Unfortunately, the test is blocked there because `app.flush()`
# needs to be called to allow the fake server to
# respond and change the status of the Deferred on which
# the client is yielding. Solution: during tests we will set
# this Flag to `False` to directly run the calls instead
# of wrapping them in `callFromThread`.
DEFERRED_RUN_IN_REACTOR_THREAD = True


class AlreadyTrainingError(Exception):
    """Raised when a training is requested for a project that is
       already training.

    Attributes:
        message -- explanation of why the request is invalid
    """

    def __init__(self):
        self.message = 'The project is already being trained!'

    def __str__(self):
        return self.message


def deferred_from_future(future):
    """Converts a concurrent.futures.Future object to a
       twisted.internet.defer.Deferred object.

    See:
    https://twistedmatrix.com/pipermail/twisted-python/2011-January/023296.html
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
    def __init__(self,
                 project_dir=None,
                 max_training_processes=1,
                 response_log=None,
                 emulation_mode=None,
                 remote_storage=None,
                 component_builder=None):
        self._training_processes = max(max_training_processes, 1)
        self.responses = self._create_query_logger(response_log)
        self.project_dir = config.make_path_absolute(project_dir)
        self.emulator = self._create_emulator(emulation_mode)
        self.remote_storage = remote_storage

        if component_builder:
            self.component_builder = component_builder
        else:
            self.component_builder = ComponentBuilder(use_cache=True)

        self.project_manager = ProjectManager.create(project_dir,
                                                     remote_storage,
                                                     component_builder)

        self.project_store = self.project_manager.get_projects()

        self.pool = ProcessPool(self._training_processes)

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
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file_name = "rasa_nlu_log-{}-{}.log".format(timestamp,
                                                            os.getpid())
            response_logfile = os.path.join(response_log, log_file_name)
            # Instantiate a standard python logger,
            # which we are going to use to log requests
            utils.create_dir_for_file(response_logfile)
            out_file = io.open(response_logfile, 'a', encoding='utf8')
            query_logger = Logger(
                    observer=jsonFileLogObserver(out_file, recordSeparator=''),
                    namespace='query-logger')
            # Prevents queries getting logged with parent logger
            # --> might log them to stdout
            logger.info("Logging requests to '{}'.".format(response_logfile))
            return query_logger
        else:
            # If the user didn't provide a logging directory, we wont log!
            logger.info("Logging of requests is disabled. "
                        "(No 'request_log' directory configured)")
            return None

    @staticmethod
    def _create_emulator(mode):
        """Create emulator for specified mode.

        If no emulator is specified, we will use the Rasa NLU format."""

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
        project = data.get("project")
        model = data.get("model")

        project_instance = self.project_manager.load_project(project)

        time = data.get('time')
        response = project_instance.parse(data['text'], time, model)

        if self.responses:
            self.responses.info('', user_input=response, project=project,
                                model=response.get('model'))

        return self.format_response(response)

    def parse_training_examples(self, examples, project, model):
        # type: (Optional[List[Message]], Text, Text) -> List[Dict[Text, Text]]
        """Parses a list of training examples to the project interpreter"""

        project_instance = self.project_manager.load_project(project)

        predictions = []
        for ex in examples:
            logger.debug("Going to parse: {}".format(ex.as_dict()))
            response = project_instance.parse(ex.text,
                                              None,
                                              model)
            logger.debug("Received response: {}".format(response))
            predictions.append(response)

        return predictions

    def format_response(self, data):
        return self.emulator.normalise_response_json(data)

    def get_status(self):
        # This will only count the trainings started from this
        # process, if run in multi worker mode, there might
        # be other trainings run in different processes we don't know about.

        return {
            "available_projects": {
                name: project.as_dict()
                for name, project in self.project_store.items()
            }
        }

    def _pre_load(self, projects):
        self.project_manager.pre_load(projects)

    def start_train_process(self,
                            data_file,  # type: Text
                            project,  # type: Text
                            train_config,  # type: RasaNLUModelConfig
                            model_name=None  # type: Optional[Text]
                            ):
        # type: (...) -> Deferred
        """Start a model training."""

        if not project:
            raise InvalidProjectError("Missing project name to train")

        project_instance = self.project_manager.load_or_create_project(project)

        if project_instance.status == 1:
            raise AlreadyTrainingError
        else:
            project_instance.status = 1

        def training_callback(model_path):
            model_dir = os.path.basename(os.path.normpath(model_path))
            project_instance.update(model_dir)
            return model_dir

        def training_errback(failure):
            logger.warning(failure)
            try:
                target_project = self.project_manager.load_project(
                    failure.value.failed_target_project)
            except InvalidProjectError:
                pass
            else:
                target_project.status = 0
            return failure

        logger.debug("New training queued")

        result = self.pool.submit(do_train_in_worker,
                                  train_config,
                                  data_file,
                                  path=self.project_dir,
                                  project=project,
                                  fixed_model_name=model_name,
                                  storage=self.remote_storage)
        result = deferred_from_future(result)
        result.addCallback(training_callback)
        result.addErrback(training_errback)

        return result

    def evaluate(self, data, project=None, model=None):
        # type: (Text, Optional[Text], Optional[Text]) -> Dict[Text, Any]
        """Perform a model evaluation."""

        project = project or RasaNLUModelConfig.DEFAULT_PROJECT_NAME
        model = model or None
        file_name = utils.create_temporary_file(data, "_training_data")
        test_data = load_data(file_name)

        preds_json = self.parse_training_examples(test_data.intent_examples,
                                                  project,
                                                  model)

        predictions = [
            {"text": e.text,
             "intent": e.data.get("intent"),
             "predicted": p.get("intent", {}).get("name"),
             "confidence": p.get("intent", {}).get("confidence")}
            for e, p in zip(test_data.intent_examples, preds_json)
        ]

        y_true = [e.data.get("intent") for e in test_data.intent_examples]
        y_true = clean_intent_labels(y_true)

        y_pred = [p.get("intent", {}).get("name") for p in preds_json]
        y_pred = clean_intent_labels(y_pred)

        report, precision, f1, accuracy = get_evaluation_metrics(y_true,
                                                                 y_pred)

        return {
            "intent_evaluation": {
                "report": report,
                "predictions": predictions,
                "precision": precision,
                "f1_score": f1,
                "accuracy": accuracy}
        }

    def unload_model(self, project, model):
        # type: (Text, Text) -> Dict[Text]
        """Unload a model from server memory."""

        if project is None:
            raise InvalidProjectError("No project specified".format(project))

        if not self.project_manager.project_exists(project):
            raise InvalidProjectError("Project {} could not "
                                      "be found".format(project))

        project_instance = self.project_manager.load_project(project)

        try:
            unloaded_model = project_instance.unload(model)
            return unloaded_model
        except KeyError:
            raise InvalidProjectError("Failed to unload model {} "
                                      "for project {}.".format(model, project))
