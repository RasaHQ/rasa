import json
from mitie import *
import os


data_dir = "../data/"
fe_file = data_dir + "total_word_feature_extractor.dat"
training_data_file = data_dir + 'intents.json'
output_file="text_categorizer.dat"


if (not os.path.exists(data_dir)):
    raise ValueError("could not find data dir, please adjust path")

# create trainer and load data
trainer = text_categorizer_trainer(fe_file)
intents_data = json.loads(open(training_data_file).read())
trainer.num_threads = 8

# add data to trainer
for ut in intents_data["utterances"]:
    tokens = tokenize(ut["text"])
    trainer.add_labeled_text(tokens,ut["intent"])

# train and save model
cat = trainer.train()
cat.save_to_disk(output_file,pure_model=True)


