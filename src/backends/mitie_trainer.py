from mitie import *
import os
import datetime


class MITIETrainer(object):
    def __init__(self):
        self.name="mitie"
        self.training_data = None
        self.intent_classifier = None
        self.entity_extractor = None
        self.training_data = None
    
    def train(self,data):
        self.training_data = data
        self.intent_classifier = self.train_intent_classifier(data.intent_examples)
        self.entity_extractor = self.train_entity_extractor(data.entity_examples)
                
    def train_entity_extractor(self,entity_examples):
        return None
    
    def train_intent_classifier(self,intent_examples):
        trainer = text_categorizer_trainer(fe_file)
        for example in intent_examples:
            tokens = tokenize(example["text"])
            trainer.add_labeled_text(tokens,example["intent"])            
        intent_classifier = trainer.train()
        return intent_classifier

        
    def persist(self,path):
        tstamp = datetime.datetime.now().strftime('%Y%m%d-%H%M')
        dirname = os.path.join(path,"model_"+tstamp)
        os.mkdir(dirname)
        data_file = os.path.join(path,"training_data.json")
        classifier_file = os.path.join(path,"intent_classifier.dat")
        entity_extractor_file = os.path.join(path,"entity_extractor.dat")
        
        metadata = {
          "trained_at":tstamp,
          "training_data":data_file,
          "backend":self.name,
          "intent_classifier":classifier_file,
          "entity_extractor": entity_extractor_file
        }
        
        with open(os.path.join(path,'metadata.json'),'w') as f:
            f.write(json.dumps(metadata,indent=4))
        with open(data_file,'w') as f:
            f.write(json.dumps(self.training_data,indent=4))

        self.intent_classifier.save_to_disk(classifier_file)
        self.entity_extractor.save_to_disk(entity_extractor_file)
        
        
        
        
        
