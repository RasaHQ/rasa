import urlparse
import multiprocessing
import os
import json
import glob
import logging
import warnings

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.train import do_train


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
        self.logfile = config.write
        self.responses = set()
        self.train_proc = None
        self.model_dir = config.path
        self.token = config.token
        self.emulator = self.__create_emulator()
        self.model_store = self.__create_model_store()

    def __featurizer_for_model(self, model):
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
            model.interpreter = self.__create_interpreter(model.metadata, model.model_dir)
            cache_key = model.model_group()
            if cache_key in cache:
                model.nlp = cache[cache_key]['nlp']
                model.featurizer = cache[cache_key]['featurizer']
            else:
                model.nlp = self.__nlp_for_backend(model.backend_name(), model.language_name())
                model.featurizer = self.__featurizer_for_model(model)
                cache[cache_key] = {'nlp': model.nlp, 'featurizer': model.featurizer}

            model_store[alias] = model
        return model_store

    def __default_model_metadata(self):
        return {
            "backend": None,
            "language_name": None,
            "feature_extractor": None
        }

    def __read_model_metadata(self, model_dir):
        if model_dir is None:
            return self.__default_model_metadata()
        else:
            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                try:
                    from rasa_nlu.persistor import Persistor
                    p = Persistor(self.config.path, self.config.aws_region, self.config.bucket_name)
                    p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
                except Exception:
                    warnings.warn("Using default interpreter, couldn't find model dir or fetch it from S3")

            with open(os.path.join(model_dir, 'metadata.json'), 'rb') as f:
                return json.loads(f.read().decode('utf-8'))

    def __create_interpreter(self, metadata, model_dir):
        backend = metadata.get("backend")
        if backend is None:
            from interpreters.simple_interpreter import HelloGoodbyeInterpreter
            return HelloGoodbyeInterpreter()
        elif backend.lower() == 'mitie':
            logging.info("using mitie backend")
            from interpreters.mitie_interpreter import MITIEInterpreter
            return MITIEInterpreter(intent_classifier=metadata['intent_classifier'], entity_extractor=metadata['entity_extractor'])
        elif backend.lower() == 'spacy_sklearn':
            logging.info("using spacy + sklearn backend")
            from interpreters.spacy_sklearn_interpreter import SpacySklearnInterpreter
            return SpacySklearnInterpreter(model_dir, **metadata)
        else:
            raise ValueError("unknown backend : {0}".format(backend))

    def __create_emulator(self):
        mode = self.config.emulate
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
        result = model.interpreter.parse(data['text'], model.nlp, model.featurizer)
        self.responses.add(json.dumps(result, sort_keys=True))
        return result

    def format(self, data):
        return self.emulator.normalise_response_json(data)

    def write_logs(self):
        with open(self.logfile, 'w') as f:
            responses = [json.loads(r) for r in self.responses]
            f.write(json.dumps(responses, indent=2))

    def get_status(self):
        if self.train_proc is not None:
            training = self.train_proc.is_alive()
        else:
            training = False

        models = glob.glob(os.path.join(self.model_dir, 'model*'))
        return json.dumps({
            "training": training,
            "available_models": models
        })

    def auth(self, path):

        if self.token is None:
            return True
        else:
            parsed_path = urlparse.urlparse(path)
            data = urlparse.parse_qs(parsed_path.query)
            valid = ("token" in data and data["token"][0] == self.token)
            return valid

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
