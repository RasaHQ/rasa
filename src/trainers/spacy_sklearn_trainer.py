import spacy
import os, datetime, json
import cloudpickle
from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier
from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from training_utils import write_training_metadata


class SpacySklearnTrainer(object):
    def __init__(self, config, language_name):
        self.name="spacy_sklearn"
        self.language_name = language_name
        self.training_data = None
        self.nlp = spacy.load(self.language_name)

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
        ner_dir = os.path.join(dirname,'ner')
        os.mkdir(ner_dir)
        entity_extractor_config_file = os.path.join(ner_dir,"config.json")
        entity_extractor_file = os.path.join(ner_dir,"model")

        write_training_metadata(dirname, tstamp, data_file, self.name, self.language_name,
                                classifier_file, ner_dir)

        with open(data_file,'w') as f:
            f.write(self.training_data.as_json(indent=2))
        with open(classifier_file,'w') as f:
            cloudpickle.dump(self.intent_classifier,f)
        with open(entity_extractor_config_file,'w') as f:
            json.dump(self.entity_extractor.ner.cfg, f)
            
        self.entity_extractor.ner.model.dump(entity_extractor_file)
