import json
from mitie import *
import config

output_file="text_categorizer.dat"

# load data and add it to trainer
intents_data = json.loads(open('intents.json').read())
for ut in intents_data["utterances"]:
    tokens = tokenize(ut["text"])
    trainer.add_labeled_text(tokens,ut["intent"])

trainer.num_threads = 8
cat = trainer.train()
cat.save_to_disk(output_file,pure_model=True)


