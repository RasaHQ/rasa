import sys, os, json
from mitie import *
from recipe_search import config
from recipe_search.recipe_domain import RecipeDomain
from recipe_search.persist import Persistor

temp_filename="temp_text_categorizer.dat"


def save_model(cat):
    cat.save_to_disk(temp_filename,pure_model=True)
    wts = config.rino.Object(**{
      "domain":'recipes01',
      "filetype":'intent_classifier',
      "equivalence_key":'recipes01-intent_classifier',
      "filename_template":"intent_classifier_{0:07d}.dat"
    })
    _ = persist.save_versioned_object(wts,temp_filename,config.RINO_NLP_MODELS)



trainer = text_categorizer_trainer(config.FE_FILEPATH)

recipe_json = json.loads(open('recipes-0.0.1.json').read())

for ut in recipe_json["utterances"]:
    tokens = tokenize(ut["text"])
    trainer.add_labeled_text(tokens,ut["intent"])


recipe_json = json.loads(open('recipes_extended.json').read())

for ut in recipe_json["utterances"]:
    tokens = tokenize(ut["text"])
    trainer.add_labeled_text(tokens,ut["intent"])

trainer.num_threads = 8
exit(0)
cat = trainer.train()

save_model(cat)

