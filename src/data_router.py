import glob
import json
import logging
import multiprocessing
import os
import tempfile
import warnings

from flask import json

from config import RasaNLUConfig
from rasa_nlu.train import do_train
from rasa_nlu.utils import mitie
from rasa_nlu.utils import spacy


class LoadedModel(object):
    def __init__(self, metadata, model_dir, interpreter=None, nlp=None):
        self.metadata = metadata
        self.model_dir = model_dir
        self.interpreter = interpreter
        self.nlp = nlp

    def backend_name(self):
        return self.metadata.get('backend')

    def language_name(self):
        return self.metadata.get('language_name')

    def featurizer_path(self):
        return self.metadata.get('feature_extractor')

    def model_group(self):
        """Groups models by backend and language name."""
        if self.language_name() is not None and self.backend_name() is not None:
            return self.language_name() + "@" + self.backend_name()
        return None


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
        ch = logging.FileHandler(path)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.propagate = False  # Prevents queries getting logged with parent logger which might log them to stdout
        logger.addHandler(ch)
        return logger

    @staticmethod
    def featurizer_for_model(model):
        """Initialize featurizer for backends. If it can NOT be shared between models, `None` should be returned."""

        if model.backend_name() == 'spacy_sklearn':
            from featurizers.spacy_featurizer import SpacyFeaturizer
            return SpacyFeaturizer()
        elif model.backend_name() == 'mitie':
            from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
            return MITIEFeaturizer(model.featurizer_path())
        else:
            return None

    def __nlp_for_backend(self, backend, language):
        """Initialize nlp for backends. If nlp can NOT be shared between models, `None` should be returned."""

        if 'spacy_sklearn' == backend:
            # spacy can share large memory objects across models
            import spacy
            return spacy.load(language, parser=False, entity=False, matcher=False)
        else:
            return None

    def __create_model_store(self):
        model_dict = self.config.server_model_dir

        # Fallback for users that specified the model path as a string and hence only want a single default model.
        if type(model_dict) is unicode or model_dict is None:
            model_dict = {"default": model_dict}

        # Reuse nlp and featurizers where possible to save memory
        cache = {}

        model_store = {}

        for alias, path in model_dict.items():
            model = LoadedModel(self.__read_model_metadata(model_dict[alias]), model_dict[alias])
            cache_key = model.model_group()
            if cache_key in cache:
                model.nlp = cache[cache_key]['nlp']
                model.featurizer = cache[cache_key]['featurizer']
            else:
                model.nlp = self.__nlp_for_backend(model.backend_name(), model.language_name())
                model.featurizer = DataRouter.featurizer_for_model(model)
                cache[cache_key] = {'nlp': model.nlp, 'featurizer': model.featurizer}
            model.interpreter = DataRouter.create_interpreter(model.nlp, model.metadata, model.model_dir)
            model_store[alias] = model
        return model_store

    def __default_model_metadata(self):
        return {
            "backend": None,
            "language_name": None,
            "feature_extractor": None
        }

    def __load_model_from_s3(self, model_dir):
        try:
            from rasa_nlu.persistor import Persistor
            p = Persistor(self.config['path'], self.config['aws_region'], self.config['bucket_name'])
            p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
        except Exception as e:
            logging.warn("Using default interpreter, couldn't fetch model: {}".format(e.message))

    def __read_model_metadata(self, model_dir):
        if model_dir is None:
            return self.__default_model_metadata()
        else:
            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                self.__load_model_from_s3(model_dir)

            with open(os.path.join(model_dir, 'metadata.json'), 'rb') as f:
                return json.loads(f.read().decode('utf-8'))

    @staticmethod
    def create_interpreter(nlp, metadata, model_dir):
        backend = metadata.get("backend")
        if backend is None:
            from interpreters.simple_interpreter import HelloGoodbyeInterpreter
            return HelloGoodbyeInterpreter()
        elif backend.lower() == mitie.MITIE_BACKEND_NAME:
            logging.info("using mitie backend")
            from interpreters.mitie_interpreter import MITIEInterpreter
            return MITIEInterpreter(intent_classifier=metadata['intent_classifier'], entity_extractor=metadata['entity_extractor'])
        elif backend.lower() == mitie.MITIE_SKLEARN_BACKEND_NAME:
            logging.info("using mitie_sklearn backend")
            from interpreters.mitie_sklearn_interpreter import MITIESklearnInterpreter
            return MITIESklearnInterpreter(**metadata)

        elif backend.lower() == spacy.SPACY_BACKEND_NAME:
            logging.info("using spacy + sklearn backend")
            from interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
            return SpacySklearnInterpreter(model_dir, nlp=nlp, **metadata)
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
            return {"error": "no model found with alias: {0}".format(alias)}
        model = self.model_store[alias]
        response = model.interpreter.parse(data['text'], nlp=model.nlp, featurizer=model.featurizer)
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
