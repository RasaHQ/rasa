import argparse
from training_data import TrainingData
import json

def update_config(config,args):
    "override config params with cmd line args, raise err if undefined"
    _args = dict(vars(args))
    for param in ["backend","data","path"]:
        replace = (_args.get(param) is not None)
        if (replace):
            config[param] = _args[param]
        if (config.get(param) is None):
            raise ValueError("parameter {0} unspecified. Please provide a value via the command line or in the config file.".format(param))

    return config

def create_argparser():
    parser = argparse.ArgumentParser(description='train a custom language parser')
    # TODO add args for training only entity extractor or only intent
    parser.add_argument('-b','--backend', default=None, choices=['mitie','sklearn'],help='which backend to use to interpret text (default: None i.e. use built in keyword matcher).')
    parser.add_argument('-p','--path', default=None, help="path where to save model files")
    parser.add_argument('-d','--data', default=None, help="file containing training data")
    parser.add_argument('-c','--config', required=True, help="config file")    
    return parser
    
def create_trainer(config):
    backend = config["backend"].lower()
    if (backend == 'mitie'):
        from backends.mitie_trainer import MITIETrainer
        return MITIETrainer(config['backends']['mitie'])
    else:
        raise NotImplementedError("other backend trainers not implemented yet")

def init():
    parser = create_argparser()
    args = parser.parse_args()
    config = json.loads(open(args.config,'rb').read())
    config = update_config(config,args)
    return config

def do_train(config):
    trainer = create_trainer(config)
    training_data = TrainingData(config["data"])
    trainer.train(training_data)
    trainer.persist(config["path"])
            


config = init()
exit(0)
do_train(config)
print("done")

