import argparse
import logging
import os

from rasa_nlu.pipeline import Trainer

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.training_data import TrainingData


def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')

    # TODO add args for training only entity extractor or only intent
    parser.add_argument('-b', '--backend', default=None, choices=['mitie', 'spacy_sklearn', 'keyword'],
                        help='backend to use to interpret text (default: built in keyword matcher).')
    parser.add_argument('-p', '--path', default=None, help="path where model files will be saved")
    parser.add_argument('-d', '--data', default=None, help="file containing training data")
    parser.add_argument('-c', '--config', required=True, help="config file")
    parser.add_argument('-l', '--language', default=None, choices=['de', 'en'], help="model and data language")
    parser.add_argument('-t', '--num_threads', default=1, type=int,
                        help="number of threads to use during model training")
    parser.add_argument('-m', '--mitie_file', default=None,
                        help='file with mitie total_word_feature_extractor')
    return parser


def create_persistor(config):
    persistor = None
    if "bucket_name" in config:
        from rasa_nlu.persistor import Persistor
        persistor = Persistor(config.path, config.aws_region, config.bucket_name)

    return persistor


def init():
    parser = create_argparser()
    args = parser.parse_args()
    config = RasaNLUConfig(args.config, os.environ, vars(args))
    return config


def do_train(config):
    """Loads the trainer and the data and runs the training of the specified model."""

    trainer = Trainer(config)

    persistor = create_persistor(config)

    training_data = TrainingData(config.data)
    trainer.validate()
    trainer.train(training_data)
    persisted_path = trainer.persist(config.path, persistor)
    return trainer, persisted_path


if __name__ == '__main__':

    config = init()
    logging.basicConfig(level=config['log_level'])

    do_train(config)
    logging.info("done")
