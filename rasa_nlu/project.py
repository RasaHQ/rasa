from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import glob

import os
import logging

from builtins import object
from threading import Lock

from typing import Text, List

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Metadata, Interpreter

logger = logging.getLogger(__name__)

MODEL_NAME_PREFIX = "model_"

FALLBACK_MODEL_NAME = "fallback"


class Project(object):
    def __init__(self, config=None, component_builder=None, project=None):
        self._config = config
        self._component_builder = component_builder
        self._models = {}
        self.status = 0
        self._reader_lock = Lock()
        self._loader_lock = Lock()
        self._writer_lock = Lock()
        self._readers_count = 0
        self._path = None
        self._project = project

        if project:
            self._path = os.path.join(self._config['path'], project)
        self._search_for_models()

    def _begin_read(self):
        # Readers-writer lock basic double mutex implementation
        self._reader_lock.acquire()
        self._readers_count += 1
        if self._readers_count == 1:
            self._writer_lock.acquire()
        self._reader_lock.release()

    def _end_read(self):
        self._reader_lock.acquire()
        self._readers_count -= 1
        if self._readers_count == 0:
            self._writer_lock.release()
        self._reader_lock.release()

    def parse(self, text, time=None, model_name=None):
        self._begin_read()

        # Lazy model loading
        if not model_name or model_name not in self._models:
            model_name = self._latest_project_model()
            if model_name not in self._models:
                logger.warn("Invalid model requested. Using default")
            else:
                logger.debug("No model specified. Using default")

        self._loader_lock.acquire()
        try:
            if not self._models.get(model_name):
                self._models[model_name] = self._interpreter_for_model(model_name)
        finally:
            self._loader_lock.release()

        response = self._models[model_name].parse(text, time)

        self._end_read()

        return response, model_name

    def update(self, model_name):
        self._writer_lock.acquire()
        self._models[model_name] = None
        self._writer_lock.release()
        self.status = 0

    def unload(self, model_name):
        self._writer_lock.acquire()
        unloaded_model = self._models.pop(model_name)
        self._writer_lock.release()
        return unloaded_model

    def _latest_project_model(self):
        """Retrieves the latest trained model for an project"""

        models = {model[len(MODEL_NAME_PREFIX):]: model
                  for model in self._models.keys()
                  if model.startswith(MODEL_NAME_PREFIX)}
        if models:
            time_list = [datetime.datetime.strptime(time, '%Y%m%d-%H%M%S')
                         for time, model in models.items()]
            return models[max(time_list).strftime('%Y%m%d-%H%M%S')]
        else:
            return FALLBACK_MODEL_NAME

    def _fallback_model(self):
        meta = Metadata({"pipeline": ["intent_classifier_keyword"]}, "")
        return Interpreter.create(meta, self._config, self._component_builder)

    def _search_for_models(self):
        model_names = (self._list_models_in_dir(self._path) +
                       self._list_models_in_cloud(self._config))
        if not model_names:
            if FALLBACK_MODEL_NAME not in self._models:
                self._models[FALLBACK_MODEL_NAME] = self._fallback_model()
        else:
            for model in set(model_names):
                if model not in self._models:
                    self._models[model] = None

    def _interpreter_for_model(self, model_name):
        metadata = self._read_model_metadata(model_name)
        return Interpreter.create(metadata, self._config,
                                  self._component_builder)

    def _read_model_metadata(self, model_name):
        if model_name is None:
            data = Project._default_model_metadata()
            return Metadata(data, model_name)
        else:
            if not os.path.isabs(model_name) and self._path:
                path = os.path.join(self._path, model_name)
            else:
                path = model_name

            # download model from cloud storage if needed and possible
            if not os.path.isdir(path):
                self._load_model_from_cloud(model_name, path, self._config)

            return Metadata.load(path)

    def as_dict(self):
        return {'status': 'training' if self.status else 'ready',
                'available_models': list(self._models.keys())}

    def _list_models_in_cloud(self, config):
        # type: (RasaNLUConfig) -> List[Text]

        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(config)
            if p is not None:
                return p.list_models(self._project)
            else:
                return []
        except Exception as e:
            logger.warn("Failed to list models of project {}. "
                        "{}".format(self._project, e))
            return []

    def _load_model_from_cloud(self, model_name, target_path, config):
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(config)
            if p is not None:
                p.retrieve(model_name, self._project, target_path)
            else:
                raise RuntimeError("Unable to initialize persistor")
        except Exception as e:
            logger.warn("Using default interpreter, couldn't fetch "
                        "model: {}".format(e))

    @staticmethod
    def _default_model_metadata():
        return {
            "language": None,
        }

    @staticmethod
    def _list_models_in_dir(path):
        if not path or not os.path.isdir(path):
            return []
        else:
            return [os.path.relpath(model, path)
                    for model in glob.glob(os.path.join(path, '*'))
                    if os.path.isdir(model)]
