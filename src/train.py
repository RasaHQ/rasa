import argparse
from training_data import TrainingData

def create_trainer(backend):
    if (backend.lower() == 'mitie'):
        from backends.mitie_trainer import MITIETrainer
        return MITIETrainer()
    else:
        raise NotImplementedError("other backend trainers not implemented yet")

parser = argparse.ArgumentParser(description='parse incoming text')
parser.add_argument('--backend', default=None, choices=['mitie','sklearn'],help='which backend to use to interpret text (default: None i.e. use built in keyword matcher).')
parser.add_argument('--path', default=None,help="path where to save model files")
parser.add_argument('--data', default=None,help="file containing training data")
# TODO add args for training only entity extractor or only intent


args = parser.parse_args()

trainer = create_trainer(args.backend)
training_data = TrainingData(args.data)
trainer.train(training_data)
trainer.persist(args.path)

print("done")

