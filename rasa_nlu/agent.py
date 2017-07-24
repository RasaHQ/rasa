from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime

import errno
import os
import logging

from builtins import object

from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu.utils import create_dir

logger = logging.getLogger(__name__)


class Agent(object):
    def __init__(self, config=None, component_builder=None, agent=None):
        self._config = config
        self._component_builder = component_builder
        self._default_model = ''
        self._models = {}
        self.status = 0

        if agent:
            self._path = os.path.join(self._config['path'], agent)
            self.update()
        else:
            meta = Metadata({"pipeline": ["intent_classifier_keyword"]}, "")
            interpreter = Interpreter.load(meta, self._config, self._component_builder)
            self._default_model = 'fallback'
            self._models['fallback'] = interpreter

    def parse(self, text, time=None, model=None):
        # Lazy model loading
        if not model:
            model = self._default_model
        if self._models[model]:
            return self._models[model].parse(text, time)
        else:
            try:
                self._models[model] = self._interpreter_for_model(model)
            except Exception as e:
                logger.warn("Using default interpreter, couldn't fetch model: {}".format(e))
                return self._models[self._default_model].parse(text, time)
            return self._models[model].parse(text, time)

    def update(self):
        last_model = self._default_model

        self._search_for_models()
        self._default_model = self._latest_agent_model()
        if last_model != self._default_model:
            self.status = 0

    def _latest_agent_model(self):
        """Retrieves the latest trained model for an agent"""
        prefix = 'model_'
        models = {model[len(prefix):]: model for model in self._models.keys()}
        if models:
            time_list = [datetime.datetime.strptime(time, '%Y%m%d-%H%M%S') for time, model in models.items()]
            return models[max(time_list).strftime('%Y%m%d-%H%M%S')]
        else:
            return None

    def _search_for_models(self):
        prefix = 'model_'

        models = {model: None for model in os.listdir(self._path) if model.startswith(prefix)}
        models.update(self._models)
        self._models = models

    def _interpreter_for_model(self, model):
        metadata = self._read_model_metadata(model)
        return Interpreter.load(metadata, self._config, self._component_builder)

    def _read_model_metadata(self, model_dir):
        if model_dir is None:
            data = Agent._default_model_metadata()
            return Metadata(data, model_dir)
        else:
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(self._path, model_dir)

            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                Agent._load_model_from_cloud(model_dir, self._config)

            return Metadata.load(model_dir)

    @staticmethod
    def _default_model_metadata():
        return {
            "language": None,
        }

    @staticmethod
    def _load_model_from_cloud(model_dir, config):
        try:
            from rasa_nlu.persistor import get_persistor
            p = get_persistor(config)
            if p is not None:
                p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
            else:
                raise RuntimeError("Unable to initialize persistor")
        except Exception as e:
            logger.warn("Using default interpreter, couldn't fetch model: {}".format(e))
