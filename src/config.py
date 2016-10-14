import os
debug_mode=True


data_dir = os.environ.get("PARSA_DATA")
fe_file= data_dir + "/total_word_feature_extractor.dat"
ner_file = data_dir + "/ner.dat"
classifier_file = data_dir + "/intent_classifier.dat"

#RECIPES
self_port=5002

