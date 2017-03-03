from rasa_nlu.classifiers.sklearn_intent_classifier import SklearnIntentClassifier


def train_intent_classifier(intent_examples, featurizer, max_num_threads, test_split_size=0.1):
    """Trains an intent classifier using sklearn.

    :param intent_examples: examples to train on
    :param featurizer: featurizer
    :param max_num_threads: number of threads to use
    :param test_split_size: how many examples to retain for testing
    :type intent_examples: list
    :type featurizer: Featurizer
    :return: trained classifier
    :rtype: SklearnIntentClassifier
    """

    intent_classifier = SklearnIntentClassifier(max_num_threads=max_num_threads)
    labels = [e["intent"] for e in intent_examples]
    sentences = [e["text"] for e in intent_examples]

    if len(set(labels)) < 2:
        raise Exception("Can not train an intent classifier. Need at least 2 different classes.")
    y = intent_classifier.transform_labels_str2num(labels)
    X = featurizer.features_for_sentences(sentences)
    intent_classifier.train(X, y, test_split_size)

    return intent_classifier
