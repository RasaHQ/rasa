from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.training_data import TrainingData


def test_classifier_regex_no_intent():
    from rasa_nlu.classifiers.regex_intent_classifier import RegExIntentClassifier
    regex_dict = {u'[0-9]+': u'provide_number',
                  u'\\bhey*': u'greet'}
    txt = "find me indian food!"
    ext = RegExIntentClassifier(regex_dict)
    assert ext.find_pattern_match(txt) is None, "No regexp from the dict matches the input"


def test_classifier_regex_intent():
    from rasa_nlu.classifiers.regex_intent_classifier import RegExIntentClassifier
    regex_dict = {u'[0-9]+': u'provide_number',
                  u'\\bhey*': u'greet'}
    txt = "heyy there!"
    ext = RegExIntentClassifier(regex_dict)
    assert ext.find_pattern_match(txt) == "greet", "Intent should be 'greet'"
