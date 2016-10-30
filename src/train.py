import argparse
from training_data import TrainingData
import json

def create_trainer(backend,config):
    if (backend.lower() == 'mitie'):
        from backends.mitie_trainer import MITIETrainer
        return MITIETrainer(config['backends']['mitie'])
    else:
        raise NotImplementedError("other backend trainers not implemented yet")

parser = argparse.ArgumentParser(description='train a custom language parser')
parser.add_argument('-b','--backend', required=True, choices=['mitie','sklearn'],help='which backend to use to interpret text (default: None i.e. use built in keyword matcher).')
parser.add_argument('-p','--path', required=True,help="path where to save model files")
parser.add_argument('-d','--data', required=True,help="file containing training data")
parser.add_argument('-c','--config', required=True,help="config file")
# TODO add args for training only entity extractor or only intent


args = parser.parse_args()

config = json.loads(open(args.config,'rb').read())
trainer = create_trainer(args.backend,config)
training_data = TrainingData(args.data)
trainer.train(training_data)
trainer.persist(args.path)

print("done")

