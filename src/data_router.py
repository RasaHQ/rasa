import urlparse
import multiprocessing
import os
import json
import glob
import logging
import warnings
from rasa_nlu.train import do_train


class DataRouter(object):
    def __init__(self, config):
        self.config = config
        self.logfile = config.write
        self.responses = set()
        self.train_proc = None
        self.interpreter_store = None
        self.nlp = None
        self.featurizer = None
        self.model_dir = config.path
        self.token = config.token
        self.emulator = self.__create_emulator()
        self.interpreter_store = self.__create_interpreter_store()

    def __create_interpreter_store(self):
        model_dict = self.config.server_model_dir
        if model_dict is None:
            return {"default": {
                     "metadata": None,
                     "interpreter": self.__create_interpreter({'backend': None}, None)
                   }}
        elif type(model_dict) is unicode:
            model_dict = {"default": model_dict}
        aliases = model_dict.keys()
        interpreter_store = {
            alias: {'metadata': self.__read_model_metadata(model_dict[alias])}
            for alias in aliases}
        backends = set([md['metadata']['backend'] for md in interpreter_store.values()])
        languages = set([md['metadata']['language_name'] for md in interpreter_store.values()])
        fe_files = set([md['metadata']['feature_extractor'] for md in interpreter_store.values()])

        assert len(languages) == 1, "models are not all in the same language, this is not supported"
        assert len(backends) == 1, "models with different backends cannot be run by same server"
        assert len(fe_files) == 1, "models with different feature extractors cannot be run by same server"

        # for spacy can share large memory objects across models
        if 'spacy_sklearn' in backends:
            import spacy
            from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
            self.nlp = spacy.load(languages.pop(), parser=False, entity=False, matcher=False)
            self.featurizer = SpacyFeaturizer()

        elif 'mitie' in backends:
            from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer
            self.featurizer = MITIEFeaturizer(list(fe_files)[0])

        for alias in aliases:
            interpreter_store[alias]['interpreter'] = self.__create_interpreter(
                interpreter_store[alias]['metadata'],
                model_dict[alias])

        return interpreter_store

    def __read_model_metadata(self, model_dir):
        metadata = None

        if model_dir is not None:
            # download model from S3 if needed
            if not os.path.isdir(model_dir):
                try:
                    from rasa_nlu.persistor import Persistor
                    p = Persistor(self.config.path, self.config.aws_region, self.config.bucket_name)
                    p.fetch_and_extract('{0}.tar.gz'.format(os.path.basename(model_dir)))
                except:
                    warnings.warn("using default interpreter, couldn't find model dir or fetch it from S3")

            metadata = json.loads(open(os.path.join(model_dir, 'metadata.json'), 'rb').read().decode('utf-8'))
        return metadata

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
        if alias not in self.interpreter_store:
            return {"error": "no model found with alias: {0}".format(alias)}
        result = self.interpreter_store[alias]["interpreter"].parse(
                   data['text'],
                   nlp=self.nlp,
                   featurizer=self.featurizer)
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
