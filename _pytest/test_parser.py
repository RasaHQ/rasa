from parsa.mitie_interpreter import MITIEInterpreter
from parsa import config


interpreter = MITIEInterpreter(config.classifier_file,config.ner_file,config.fe_file)



def test_samples():
    samples = [
      ("I would like some chinese food please",{'intent':'inform', 'entities' : {'cuisine': 'chinese'}}),
      ("Is that vegetarian",{'intent':'ask_constraint', 'entities' :{'diet': 'vegetarian'}})
    ]   

    for text, result in samples:
        assert interpreter.parse(text) == result, "text : {0} \nresult : {1}, expected {2}".format(text,interpreter.parse(text),result)



