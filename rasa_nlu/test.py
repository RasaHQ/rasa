from rasa_nlu.data_router import DataRouter
from rasa_nlu.pipeline import Interpreter
from train import init


if __name__ == '__main__':
    config = init()
    # persisted_path = "model_20170309-105707"
    persisted_path = "model_20170309-172517"
    metadata = DataRouter.read_model_metadata(persisted_path, config)
    interpreter = Interpreter.load(metadata, config)
    result = interpreter.parse(u"i am looking for a chines restaurant")
    print result
