import glob
import json
import logging
import multiprocessing
import os
import tempfile

from flask import json

from config import RasaNLUConfig
from rasa_nlu.train import do_train
from rasa_nlu.utils import mitie
from rasa_nlu.utils import spacy
from rasa_nlu.util import create_dir_for_file


class InvalidModelError(Exception):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message):
        self.message = message


class ModelMetadata(object):
    def __init__(self, metadata, model_dir):
        self.metadata = metadata
        self.model_dir = model_dir

    def __prepend_path(self, prop):
        if self.metadata.get(prop) is not None:
            return os.path.normpath(os.path.join(self.model_dir, self.metadata[prop]))
        else:
            return None

    @property
    def feature_extractor_path(self):
        return self.__prepend_path("feature_extractor")

    @property
    def intent_classifier_path(self):
        return self.__prepend_path("intent_classifier")

    @property
    def entity_extractor_path(self):
        return self.__prepend_path("entity_extractor")

    @property
    def entity_synonyms_path(self):
        return self.__prepend_path("entity_synonyms")

    def language_name(self):
        return self.metadata.get('language_name')

    def backend_name(self):
        return self.metadata.get('backend')

    def model_group(self):
        """Groups models by backend and language name."""
        if self.language_name() is not None and self.backend_name() is not None:
            return self.language_name() + "@" + self.backend_name()
        return None


class LoadedModel(object):
    def __init__(self, metadata, model_dir, interpreter=None, nlp=None, featurizer=None):
        self.metadata = metadata
        self.model_dir = model_dir
        self.interpreter = interpreter
        self.featurizer = featurizer
        self.nlp = nlp


class DataRouter(object):
    def __init__(self, config):
        self.config = config
        # Ensures different log files for different processes in multi worker mode
        self.logfile = config['write'].replace(".json", "-{}.json".format(os.getpid()))
        self.responses = DataRouter._create_query_logger(self.logfile)
        self.train_procs = []
        self.model_dir = config['path']
        self.token = config['token']
        self.emulator = self.__create_emulator()
        self.model_store = self.__create_model_store()

    @staticmethod
    def _create_query_logger(path):
        """Creates a logger that will persist incomming queries and their results."""

        logger = logging.getLogger('query-logger')
        logger.setLevel(logging.INFO)
        create_dir_for_file(path)
        ch = logging.FileHandler(path)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.propagate = False  # Prevents queries getting logged with parent logger which might log them to stdout
        logger.addHandler(ch)
        return logger

    @staticmethod
    def featurizer_for_model(model, nlp):
        """Initialize featurizer for backends. If it can NOT be shared between models, `None` should be returned."""

        if model.backend_name() == spacy.SPACY_BACKEND_NAME:
            from featurizers.spacy_featurizer import SpacyFeaturizer
            return SpacyFeaturizer(nlp)
        elif model.backend_name() == mitie.MITIE_BACKEND_NAME or \
                model.backend_name() == mitie.MITIE_SKLEARN_BACKEND_NAME:
            from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
            return MITIEFeaturizer(model.feature_extractor_path)
        else:
            return None

    @staticmethod
    def nlp_for_backend(backend, language):
        """Initialize nlp for backends. If nlp can NOT be shared between models, `None` should be returned."""

        if spacy.SPACY_BACKEND_NAME == backend:
            # spacy can share large memory objects across models
            import spacy as sp
            return sp.load(language, parser=False, entity=False, matcher=False)
        else:
            return None

    def __search_for_models(self):
        models = {}
        for metadata_path in glob.glob(os.path.join(self.config.path, '*/metadata.json')):
            model_name = os.path.basename(os.path.dirname(metadata_path))
            models[model_name] = model_name
        return models

    def __create_model_store(self):
        # Fallback for users that specified the model path as a string and hence only want a single default model.
        if type(self.config.server_model_dirs) is unicode or type(self.config.server_model_dirs) is str:
            model_dict = {"default": self.config.server_model_dirs}
        elif self.config.server_model_dirs is None:
            model_dict = self.__search_for_models()
        else:
            model_dict = self.config.server_model_dirs

        # Reuse nlp and featurizers where possible to save memory
        cache = {}

        model_store = {}

        for alias, model_path in model_dict.items():
            metadata = DataRouter.read_model_metadata(model_path, self.config)
            cache_key = metadata.model_group()
            if cache_key in cache:
                nlp = cache[cache_key]['nlp']
                featurizer = cache[cache_key]['featurizer']
            else:
                nlp = DataRouter.nlp_for_backend(metadata.backend_name(), metadata.language_name())
                featurizer = DataRouter.featurizer_for_model(metadata, nlp)
                cache[cache_key] = {'nlp': nlp, 'featurizer': featurizer}
            interpreter = DataRouter.create_interpreter(nlp, metadata)
            model_store[alias] = LoadedModel(metadata, model_path, interpreter, nlp, featurizer)
        return model_store

    @staticmethod
    def default_model_metadata():
        return {
            "backend": None,
            "language_name": None,
            "feature_extractor": None
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
        else:
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(config['path'], model_dir)

            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                DataRouter.load_model_from_s3(model_dir, config)

            with open(os.path.join(model_dir, 'metadata.json'), 'rb') as f:
                data = json.loads(f.read().decode('utf-8'))
        return ModelMetadata(data, model_dir)

    @staticmethod
    def create_interpreter(nlp, metadata):
        backend = metadata.backend_name()
        if backend is None:
            from interpreters.simple_interpreter import HelloGoodbyeInterpreter
            return HelloGoodbyeInterpreter()
        elif backend.lower() == mitie.MITIE_BACKEND_NAME:
            logging.info("using mitie backend")
            from interpreters.mitie_interpreter import MITIEInterpreter
            return MITIEInterpreter.load(metadata)
        elif backend.lower() == mitie.MITIE_SKLEARN_BACKEND_NAME:
            logging.info("using mitie_sklearn backend")
            from interpreters.mitie_sklearn_interpreter import MITIESklearnInterpreter
            return MITIESklearnInterpreter.load(metadata)

        elif backend.lower() == spacy.SPACY_BACKEND_NAME:
            logging.info("using spacy + sklearn backend")
            from interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
            return SpacySklearnInterpreter.load(metadata, nlp)
        else:
            raise ValueError("unknown backend : {0}".format(backend))

    def __create_emulator(self):
        mode = self.config['emulate']
        if mode is None:
            from emulators import NoEmulator
            return NoEmulator()
        elif mode.lower() == 'wit':
            from emulators.wit import WitEmulator
            return WitEmulator()
        elif mode.lower() == 'luis':
            from emulators.luis import LUISEmulator
            return LUISEmulator()
        elif mode.lower() == 'api':
            from emulators.api import ApiEmulator
            return ApiEmulator()
        else:
            raise ValueError("unknown mode : {0}".format(mode))

    def extract(self, data):
        return self.emulator.normalise_request_json(data)

    def parse(self, data):
        alias = data.get("model") or "default"
        if alias not in self.model_store:
            raise InvalidModelError("No model found with alias '{}'".format(alias))
        else:
            model = self.model_store[alias]
            response = model.interpreter.parse(data['text'])
            self.responses.info(json.dumps(response, sort_keys=True))
            return self.format_response(response)

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
        fd, fname = tempfile.mkstemp(suffix="_training_data.json")
        os.write(fd, data)
        os.close(fd)
        _config = dict(self.config.items())
        _config["data"] = fname
        train_config = RasaNLUConfig(cmdline_args=_config)
        process = multiprocessing.Process(target=do_train, args=(train_config,))
        self.train_procs.append(process)
        process.start()
        logging.info("Training process {} started".format(process))
