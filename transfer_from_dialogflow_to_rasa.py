import os
import json
import sys

def load_file_to_json(full_path):
    with open(full_path) as f:
        return json.load(f)

def harvest_examples(dialogflow_examples, intent):
    harvested_examples = list()
    for example in dialogflow_examples:
        text = ""
        entities = list()
        for block in example['data']:
            blockText = block['text']
            if "alias" in block:
                text_length = len(text)
                entities.append({
                    "start": text_length,
                    "end": text_length + len(blockText),
                    "value": blockText,
                    "entity": block['alias']
                })
            text += blockText

        harvested_example = {
            "intent": intent,
            "text": text
        }

        if(len(entities) > 0):
            harvested_example['entities'] = entities

        harvested_examples.append(harvested_example)

    return harvested_examples

def get_current_action(file):
    action_file = file.replace("_usersays_en", "")
    data_in_file = load_file_to_json(action_file)
    if 'action' in data_in_file['responses'][0]:
        return data_in_file['responses'][0]['action']
    return data_in_file['name']

def get_common_examples(intents_dir):
    files = os.listdir(intents_dir)
    common_examples = list()
    for file in files:
        data_in_file = load_file_to_json(intents_dir + "/" + file)
        if file.endswith("_usersays_en.json"):
            current_intent_action = get_current_action(intents_dir + "/" + file)
            common_examples += harvest_examples(data_in_file, current_intent_action)
    return common_examples

def get_current_entity_name(file):
    entity_file = file.replace("_entries_en", "")
    data_in_file = load_file_to_json(entity_file)
    return data_in_file['name']

def harvest_synonymn(value, synonyms):
    # RASA doesn't want entries with mirror synonymns
    # and we dont want to add composite entries
    if value in synonyms:
        synonyms.remove(value)

    if(len(synonyms) == 0 or value[0] == '@'):
        return False

    return {
        "value": value,
        "synonyms": synonyms
    }

def harvest_lookup_table(value, synonyms):
    if '@' in value:
        return False

    synonyms.append(value)
    return synonyms

def harvest_composite_entries(value, entity):
    composite_entries = list()
    value = value.strip()

    if '@' not in value:
        return False

    splitValue = value.split(" ")
    
    for each in splitValue:
      if each:
        # Because RASA removes duplicates entries so we prefix 
        # with the current entity so they get to be unique 
        # and we remove it while training
        composite_entries.append("@" + entity + "_" + each) 

    return composite_entries

def process_data_entities(dialogflow_entities, current_entity_name):
    entity_synonyms = list()
    lookup_table_entries = list()
    composite_entities = list()
    for block in dialogflow_entities:
        value = block['value']
        synonyms = block['synonyms']

        harvested_synonymn = harvest_synonymn(value, synonyms)
        if(harvested_synonymn):
            entity_synonyms.append(harvested_synonymn)

        harvested_lookup_table = harvest_lookup_table(value, synonyms)
        if(harvested_lookup_table):
            lookup_table_entries += harvested_lookup_table

        harvested_composite_entries = harvest_composite_entries(value, current_entity_name)
        if(harvested_composite_entries):
            composite_entities += harvested_composite_entries

    lookup_tables = []
    if(len(lookup_table_entries) > 0):
        lookup_tables = [{
            "name": current_entity_name,
            "entries": lookup_table_entries
        }]

    composite_entries = []

    if(len(composite_entities) > 0):
        entity_synonyms.append({
            "value": "@" + current_entity_name,
            "synonyms": list(set(composite_entities))
        })

    return {
        "entity_synonyms": entity_synonyms, 
        "lookup_tables": lookup_tables
    }

def process_entities(entities_dir):
    files = os.listdir(entities_dir)
    entity_synonyms = list()
    lookup_tables = list()
    composite_entries = list()
    for file in files:
        data_in_file = load_file_to_json(entities_dir + "/" + file)
        if file.endswith("_entries_en.json"):
            current_entity_name = get_current_entity_name(entities_dir + "/" + file)
            data = process_data_entities(data_in_file, current_entity_name)
            entity_synonyms += data.get("entity_synonyms")
            lookup_tables += data.get("lookup_tables")

    return {
        "entity_synonyms": entity_synonyms, 
        "lookup_tables": lookup_tables, 
    }

def main():
    path_arg = sys.argv[1]
    directory = path_arg if path_arg.endswith('/') else path_arg + '/'
    common_examples = get_common_examples(directory + "intents")
    entities_data = process_entities(directory + "entities")

    entity_synonyms = entities_data.get("entity_synonyms")
    lookup_tables = entities_data.get("lookup_tables")

    training_data = {
        "rasa_nlu_data": {
            "common_examples": common_examples,
            "entity_synonyms": entity_synonyms,
            "lookup_tables": lookup_tables,
            "regex_features": [],
        }
    }

    with open(directory + "training_data.json", "w") as training_data_file:
        json.dump(training_data, training_data_file, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
