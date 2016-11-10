import spacy
import os, datetime, json
import cloudpickle
from parsa.featurizers.spacy_featurizer import SpacyFeaturizer
from parsa.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from parsa.extractors.spacy_entity_extractor import SpacyEntityExtractor

class SpacySklearnTrainer(object):
    def __init__(self,config):
        self.name="spacy_sklearn"
        self.training_data = None
        self.nlp = spacy.load('en')

        self.featurizer = SpacyFeaturizer(self.nlp)
        self.intent_classifier = SklearnIntentClassifier()
        self.entity_extractor = SpacyEntityExtractor()    
        
    def train(self,data):
        self.training_data = data
        self.entity_extractor.train(self.nlp,data.entity_examples)        
        self.train_intent_classifier(data.intent_examples)
                        
    def train_intent_classifier(self,intent_examples):
        labels = [e["intent"] for e in intent_examples]
        sents = [e["text"] for e in intent_examples]
        y = self.intent_classifier.transform_labels(labels)
        X = self.featurizer.create_bow_vecs(sents)
        self.intent_classifier.train(X,y)
        
    def persist(self,path):
        tstamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        dirname = os.path.join(path,"model_"+tstamp)
        os.mkdir(dirname)
        data_file = os.path.join(dirname,"training_data.json")
        classifier_file = os.path.join(dirname,"intent_classifier.pkl")
        entity_extractor_config_file = os.path.join(dirname,"entity_extractor.json")
        entity_extractor_file = os.path.join(dirname,"entity_extractor")
        
        metadata = {
          "trained_at":tstamp,
          "training_data":data_file,
          "backend":self.name,
          "intent_classifier":classifier_file,
          "entity_extractor": entity_extractor_file,
          "entity_extractor_config" : entity_extractor_config_file
        }
        
        with open(os.path.join(dirname,'metadata.json'),'w') as f:
            f.write(json.dumps(metadata,indent=4))
        with open(data_file,'w') as f:
            f.write(self.training_data.as_json(indent=2))
        with open(classifier_file,'w') as f:
            cloudpickle.dump(self.intent_classifier,f)
        with open(entity_extractor_config_file,'w') as f:
            json.dump(self.entity_extractor.ner.cfg, f)
            
        self.entity_extractor.ner.model.dump(entity_extractor_file)
        
        
        
