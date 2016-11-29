import argparse
from training_data import TrainingData
from rasa_nlu.util import update_config
import json, warnings, os


def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')

    # TODO add args for training only entity extractor or only intent
    parser.add_argument('-b', '--backend', default='keyword', choices=['mitie', 'spacy_sklearn', 'keyword'],
                        help='backend to use to interpret text (default: built in keyword matcher).')
    parser.add_argument('-p', '--path', default=None, help="path where model files will be saved")
    parser.add_argument('-d', '--data', default=None, help="file containing training data")
    parser.add_argument('-c', '--config', required=True, help="config file")
    parser.add_argument('-l', '--language', default='en', choices=['de', 'en'], help="model and data language")
    parser.add_argument('-m','--mitie_file', default='data/total_word_feature_extractor.dat', help='file with mitie total_word_feature_extractor')    
    return parser


def create_trainer(config):
    backend = config['backend'].lower()
    if backend == 'mitie':
        from trainers.mitie_trainer import MITIETrainer
        return MITIETrainer(config['mitie_file'], config['language'])
    if backend == 'spacy_sklearn':
        from trainers.spacy_sklearn_trainer import SpacySklearnTrainer
        return SpacySklearnTrainer(config['language'])
    else:
        raise NotImplementedError("other backend trainers not implemented yet")


def load_configuration(file_name):
    config = {}
    if (os.path.isfile(file_name)):
        config = json.loads(open(file_name, 'rb').read())
    else:
        warnings.warn("could not find config file {0}, ignoring".format(file_name))
    return config

def create_persistor(config):
    persistor = None
    try:
        from rasa_nlu.persistor import Persistor
        persistor = Persistor(config['path'],config['aws_region'],config['bucket_name'])        
    except:
        pass
    return persistor


def init():
    parser = create_argparser()
    args = parser.parse_args()
    config = load_configuration(args.config)
    config = update_config(config,args,exclude=['config'],required=['path','backend','data'])

    return config


def do_train(config):
    trainer = create_trainer(config)

    persistor = create_persistor(config)
    
    training_data = TrainingData(config["data"],config["backend"], config["language"])
    trainer.train(training_data)
    trainer.persist(config["path"],persistor)
            


if __name__ == '__main__':
    config = init()
    do_train(config)
    print("done")
