import sys, os, json, re, random, string
from mitie import *
from collections import defaultdict

from recipe_search import config , ingredients
from recipe_search.recipe_domain import RecipeDomain
from recipe_search.persist import Persistor



num_training_pts=10000
temp_filename="temp_ner_model.dat"
noise_level = 0.01
letters = string.letters[:26]
persist = Persistor(config.rino_token,config.rino_dir)
fe_filepath = config.data_dir + '/total_word_feature_extractor.dat'
domain = RecipeDomain("",False)

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def add_noise(text):
    # randomly replace and swap chars
    ls = list(text)
    for i in range(len(ls)-1):
        rand = random.random()
        if (rand < noise_level /2):
            ls[i],ls[i+1] = ls[i+1],ls[i]
        elif (rand < noise_level):
            ls[i] = random.choice(letters)
    return ''.join(ls)

def save_model(ner):
    ner.save_to_disk(temp_filename,pure_model=True)
    wts = persist._rino.Object(**{
      "domain":'recipes01',
      "filetype":'named_entity_extractor',
      "equivalence_key":'recipes01-named_entity_extractor',
      "num_examples":num_training_pts,
      "filename_template":"ner_{0:07d}.dat"
    })
    _ = persist.save_versioned_object(wts,temp_filename,config.rino_nlp_models)


def random_training_entity(templates,slot_vals,sentences):

    
    text = random.choice(templates)
    intent = "ask_constraint" if text in enquire_templates else "inform"

    keywords = [s[1:-1] for s in re.findall("\{.*?\}",text)]

    entities = []

    for key in keywords:
        temp_tokens=tokenize(text)
        try:
            _start = temp_tokens.index('{'+key+'}')
        except:
            print(temp_tokens)
            print(key)
            exit(0)
        bare_key = key.split('_')[0]
        slot_val = random.choice(slot_vals[bare_key])
        slot_tokens = tokenize(slot_val)
        _end = _start + len(slot_tokens) 
        entities.append((_start,_end,bare_key))

        d=SafeDict()
        d[key]=slot_val
        text = string.Formatter().vformat(text, (), d)

    #text = add_noise(text)

    if (text in sentences):
        return None
    sentences.append(text)

    with open('utterances_ner.json','a') as f:
        f.write('{{ "intent" : "{i}", "text" : "{t}" }}, \n'.format(i=intent,t=text))

    tokens = tokenize(text)
    #print(text)
    #print(entities)
    sample = ner_training_instance(tokens)
    for ent in entities:        
        try:
            sample.add_entity(xrange(ent[0],ent[1]),ent[2])
        except:
            print(text)
            print(ent)
            exit(0)
    
    return sample




trainer = ner_trainer(fe_filepath)

diets = []
for d in ["vegetarian","vegan","glutenFree","dairyFree"]:
    for word in domain.slot_value_aliases["diet"][d]:
        diets.append(word)

effort_levels = []



# handle antonyms
# handle queries 
dish_names = [
  "shepherd's pie", "lasagne", "casserole", "pasta", "pizza", "biryani", "pie", "paella", "curry", "burrito", "taco",
  "risotto", "macaroni", "chowder", "pad thai", "samosa", "dumpling", "sandwich", "leftover", "jambalaya", "schnitzel",
  "hamburger", "salad", "stir fry", "cake", "cookie", "margarita", "cocktail", "gazpacho", "christmas", "halloween", "easter"
]
   
cuisines = ["italian","french","german","chinese","iranian","indian","american","british","mexican"]
dish_types = ["main course", "side dish", "dessert", "appetizer", "salad", "bread", "breakfast", 
              "soup", "beverage", "sauce", "drink"]
ingredients = ingredients.ingredient_list

slot_vals = {
  "ingredient" : ingredients,
  "cuisine" : cuisines,
  "type" : dish_types,
  "query" : dish_names
}

for slot in [s for s in domain.slots if not s in slot_vals]:
    ls=[]
    for val, _ls in domain.slot_value_aliases[slot].iteritems():
        ls += _ls
    slot_vals[slot] = ls
    
enquire_templates = [
  "is that {expense} ?",
  "does that have {ingredient} in it?",
  "does it contain {ingredient}",
  "is that {popular}",
  "is it {expense} ?",
  "is that a {type}",
  "how {expense} ?",
  "how {effort} ?",
  "is that {diet} ?",
  "is it {healthy}",
  "how {healthy} is that?",
  "how {effort} is it?",
  "is it {effort} to make?"
]

inform_templates = [
  "I want to make {cuisine} food",
  "{cuisine} food with {ingredient} in it",
  "show me {cuisine} recipes",
  "I want to cook something with {ingredient}",
  "{cuisine} please",
  "do you know any {effort} {query} to make?",
  "I want something {diet}",
  "show me something {healthy}",
  "show me {query} recipes",
  "I want to make a {query}",
  "show me a {popular} {query}",
  "maybe a {diet} {query}",
  "I'll make want anything {expense}",
  "whatever is {effort}",
  "it should be {effort}",
  "I want to make {query}",
  "Is that {diet}",
  "How about a {diet} and {diet} {type}",
  "Suggest a {type} which is {diet}",
  "Suggest a {cuisine} {query}",
  "{cuisine} {query}",
  "{ingredient} {query}",
  "{query} with {ingredient}",
  "a {diet} {query} with {ingredient}",
  "with {ingredient} and {ingredient_1}",
  "a {diet} {cuisine} {query} with {ingredient} {ingredient_1} and {ingredient_2}",
  "{cuisine} food with {ingredient} and {ingredient_1}",
  "i have {ingredient} and {ingredient_1}",
  "I have {ingredient} {ingredient_1} {ingredient_2} at home, what would you propose?",
  "something {cuisine} with {ingredient} and {ingredient_1}",
  "Can you recommend a typical {cuisine} dish?",
  "Hi, I really like {cuisine} food.",
  "Mh, what is your suggestion for a {type}? Something {cuisine}",
  "And for a {diet} dish?",
  "what about a {type}",
  "hmm show me a {type}",
  "how about a {ingredient} {query}?",
  "I've got {ingredient}, {ingredient_1} and {ingredient_2}.. what should I make?",
  "show me an {cuisine} {query}!",
  "show me {cuisine} {type}",
  "hi! I'd like to cook a {query}",
  "Hi, can you make me any suggestions for {type} tonight?",
  "Can you give me a recipe for {query}?",
  "{type}",
  "do you have a {diet} recipe?",
  "do you know a {query} recipe?",
  "what about a {query}?",
  "how about {cuisine}?",
  "please find me a {query}",
  "please find me an {ingredient} {query}",
  "try find a {query}",
  "can you give me the reciepe for a {query}",
  "{cuisine} {query} that is {effort}",
  "any {type}?",
  "Can you suggest a {diet} {cuisine} meal?",
  "I want to cook {query}",
  "can you show me {cuisine} food please",
  "which {type} can you recommend?",
  "do you have {query} recipes?",
  "Give me a recipe for {cuisine} {query}",
  "Can you give me a recipe which includes {ingredient}?",
  "Can you provide a {diet} {cuisine} {type}?",
  "{cuisine}",
  "hey, id like to cook a meal that can be prepared {effort}",
  "i would like to cook a {diet} meal that takes {effort} to be prepared"
]

templates  = enquire_templates + inform_templates

sample_set=[]
sentences = []
for _ in range(num_training_pts):
    sample = random_training_entity(templates,slot_vals,sentences)
    if (sample):
        sample_set.append(sample)

for sample in sample_set:    
    trainer.add(sample)


    
trainer.num_threads = 8
trainer.beta = 0.1

ner = trainer.train()
save_model(ner)







