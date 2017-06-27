from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Interpreter
from rasa_nlu.model import Metadata

if __name__ == '__main__':
    # get the metadata config from the package data
    metadata = Metadata.load("/Users/tmbo/lastmile/bot-ai/rasa_nlu/models/model_20170608-171457")
    interpreter = Interpreter.load(metadata, RasaNLUConfig("../config_spacy.json"))

    text = "aws s3 ls"
    response = interpreter.parse(text)
    print(response)
