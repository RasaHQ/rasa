import sys, os, json, re, random, string
from mitie import *
from collections import defaultdict

num_training_pts=100
temp_filename="temp_ner_model.dat"
fe_filepath = config.data_dir + '/total_word_feature_extractor.dat'

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def random_training_entity(templates,slot_vals,sentences):
    
    intent = random.choice(templates.keys())    
    text = random.choice(templates[intent])

    keywords = [s[1:-1] for s in re.findall("\{.*?\}",text)]
    entities = []

    # insert slot values into template string
    for key in keywords:
        temp_tokens=tokenize(text)
        _start = temp_tokens.index('{'+key+'}')
        bare_key = key.split('_')[0]
        slot_val = random.choice(slot_vals[bare_key])
        slot_tokens = tokenize(slot_val)
        _end = _start + len(slot_tokens) 
        entities.append((_start,_end,bare_key))

        d=SafeDict()
        d[key]=slot_val
        text = string.Formatter().vformat(text, (), d)


    if (text in sentences):
        return None
    sentences.append(text)

    with open('utterances_ner.json','a') as f:
        f.write('{{ "intent" : "{i}", "text" : "{t}" }}, \n'.format(i=intent,t=text))

    tokens = tokenize(text)

    sample = ner_training_instance(tokens)
    for ent in entities:        
        try:
            sample.add_entity(xrange(ent[0],ent[1]),ent[2])
        except:
            print(text)
            print(ent)
            exit(0)
    
    return sample


## Templates, slot values
slot_vals = {
  "cuisine":["italian","french","german","chinese","iranian","indian","american","british","mexican"],
  "expense":["cheap","pricey","expensive","not expensive"]
}

templates = {
  "inform" : [
    "show me {cuisine} spots",
    "{cuisine} please",
    "anything {expense}",
    "whatever is {expense}",
    "it should be {expense}",
    "{expense} and {cuisine}"
    "I want {cuisine} or {cuisine_1} food",
    "please find me a {cuisine} restaurant",
    "{cuisine}",
  ],
  "ask_constraint" : [
    "is that {expense}",
    "isn't that very {expense}",
    "looks {expense}?",
    "is that {cuisine}",
    "is it a {cuisine} place?"
  ]
}



## generate training data
sample_set=[]
sentences = []
for _ in range(num_training_pts):
    sample = random_training_entity(templates,slot_vals,sentences)
    if (sample):
        sample_set.append(sample)

## prepare trainer instance
trainer = ner_trainer(fe_filepath)
for sample in sample_set:    
    trainer.add(sample)

##  train model
trainer.num_threads = 8
trainer.beta = 0.1
ner = trainer.train()

## save model
ner.save_to_disk(temp_filename,pure_model=True)








