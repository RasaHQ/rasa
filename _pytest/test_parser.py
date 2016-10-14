from parsa.mitie_interpreter import MITIEInterpreter
from parsa import config

# paths
#test_data_file = config.data_dir + '/test_sentences.json'

interpreter = MITIEInterpreter(config.classifier_file,config.ner_file,config.fe_file)

samples = [
  ("I would like some chinese food please",('inform', {'cuisine': 'chinese'})),
  ("Is that vegetarian",('ask_constraint', {'diet': 'vegetarian'})),
  ("How about a gluten and lactose free pudding",('inform', {'diet': 'glutenFree','diet' : 'lactoseFree','type':'dessert'})),
  ("Suggest a dessert which is gluten and lactose free",('inform', {'diet': 'glutenFree','diet' : 'lactoseFree','type':'dessert'})),
  ("Could you suggest another",('deny',{})),
  ("Please give me a recipe for a pudding which does not include oats and is gluten free and lactose free",{})
]


for text, result in samples:
    assert interpreter.parse(text) == result, "text : {0} \nresult : {1}".format(text,interpreter.parse(text))



