import numpy as np
from mitie import *
import itertools
from parsa.training_data import TrainingData
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import pickle


def transform_labels(labels):
    le=LabelEncoder()
    y=le.fit_transform(labels)
    return y, le
    

def train(X,y):    
    tuned_parameters = [
      {'C': [1,2,5, 10,20, 100], 'kernel': ['linear']}
      #{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
     ]
    score = 'f1'#'precision'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0)

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=2,
                       scoring='%s_weighted' % score)
                                     
    clf.fit(X_train, y_train)
    

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    
    return clf


def test_clf(clf,X,le):
    y_pred = clf.predict(X)
    return le.inverse_transform(y_pred)

def create_bow_vecs(sentences,feature_extractor):
    ndim = feature_extractor.num_dimensions
    X=np.zeros((len(sentences),ndim))
    for idx, sent in enumerate(sentences):
        tokens = tokenize(sent)
        vec = np.zeros(ndim)
        for token in tokens:
            vec += feature_extractor.get_feature_vector(token)
        X[idx,:] =  vec / len(tokens)
    return X
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #print(tick_marks)
    #xlabs = 
    plt.grid(True)
    plt.xticks(tick_marks, [c[:8] for c in classes], rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if (cm[i,j] > 0.0):
            plt.text(j, i, '%1.0f'%cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
        

parsa_data = TrainingData("/Users/alan/Developer/dialog/cs-demo/data/addresses-0.2.json")
feature_extractor = total_word_feature_extractor("/Users/alan/Developer/dialog/cs-demo/data/total_word_feature_extractor.dat")

sentences = [e["text"] for e in parsa_data.intent_examples]
labels = [e["intent"] for e in parsa_data.intent_examples]

X = create_bow_vecs(sentences,feature_extractor)
y, le = transform_labels(labels)
counts = [(l,len([i for i in labels if i == l])) for l in set(labels)]

clf = train(X,y)

labels_pred =  test_clf(clf,X,le)
y_pred = le.transform(labels_pred)
cmat = confusion_matrix(y, y_pred)

np.set_printoptions(precision=1)

#plt.figure(figsize=(10,10))
#plot_confusion_matrix(cmat, classes=le.classes_, normalize=False,title='confusion matrix')
#plt.show()
                      
idx = np.arange(len(labels))
np.random.shuffle(idx)

num_errs = np.sum(np.abs(y - y_pred))
errs = np.where(y != y_pred)

print("misclassifications : ")
for i in range(len(labels)):
    if (y[i] != y_pred[i]):
        print("text : {0} pred :{1} corr {2}".format(sentences[i],labels_pred[i],labels[i]))
