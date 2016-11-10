import argparse
from training_data import TrainingData
from parsa.util import update_config
import json



def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')
    # TODO add args for training only entity extractor or only intent
    parser.add_argument('-b','--backend', default=None, choices=['mitie','sklearn'],help='which backend to use to interpret text (default: None i.e. use built in keyword matcher).')
    parser.add_argument('-p','--path', default=None, help="path where model files will be saved")
    parser.add_argument('-d','--data', default=None, help="file containing training data")
    parser.add_argument('-c','--config', required=True, help="config file")    
    return parser
    
def create_trainer(config):
    backend = config["backend"].lower()
    if (backend == 'mitie'):
        from trainers.mitie_trainer import MITIETrainer
        return MITIETrainer(config['backends']['mitie'])
    if (backend == 'spacy_sklearn'):
        from trainers.spacy_sklearn_trainer import SpacySklearnTrainer
        return SpacySklearnTrainer(config['backends']['spacy_sklearn'])    
    else:
        raise NotImplementedError("other backend trainers not implemented yet")

def init():
    parser = create_argparser()
    args = parser.parse_args()
    config = json.loads(open(args.config,'rb').read())
    config = update_config(config,args,exclude=['config'],required=['path','backend','data'])
    return config

def do_train(config):
    trainer = create_trainer(config)
    training_data = TrainingData(config["data"])
    trainer.train(training_data)
    trainer.persist(config["path"])
            


config = init()
do_train(config)
print("done")

