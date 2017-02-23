from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier


def train_intent_classifier(intent_examples, featurizer, max_num_threads, test_split_size=0.1, nlp=None):
    intent_classifier = SklearnIntentClassifier(max_num_threads=max_num_threads)
    labels = [e["intent"] for e in intent_examples]
    sentences = [e["text"] for e in intent_examples]
    y = intent_classifier.transform_labels_str2num(labels)
    X = featurizer.create_bow_vecs(sentences, nlp)
    intent_classifier.train(X, y, test_split_size)

    return intent_classifier
