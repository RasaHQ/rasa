import os
debug_mode=True

#root_dir="/Users/alan/Developer/dialog/parsa"
root_dir=os.environ["PARSA_DATA"] or  os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#"/Users/alan/Developer/dialog/lastmile/framework/rasa/examples/recipes"
data_dir = root_dir + "/data"
fe_file= data_dir + "/total_word_feature_extractor.dat"
ner_file = data_dir + "/ner.dat"
classifier_file = data_dir + "/intent_classifier.dat "

#RECIPES
self_port=5002

