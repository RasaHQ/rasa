import json
import os
import pytest
from rasa_nlu.test import collect_ner_results


# read generated json file and return the number of tp, fp and fn
def analyze_file(ner_filename):
    with open(ner_filename, 'r', encoding='utf-8') as fs:
        data = json.load(fs)
    tp_count = len(data['TP'])
    fp_count = len(data['FP'])
    fn_count = len(data['FN'])
    return (tp_count, fp_count, fn_count)


def test_collect_ner_results_both_empty():
    utterance_targets = []
    utterance_predictions = []
    ner_filename = 'test.json'
    collect_ner_results(utterance_targets, utterance_predictions, ner_filename)
    (tp, fp, fn) = analyze_file(ner_filename)
    os.remove(ner_filename)
    assert(tp == 0)
    assert(fp == 0)
    assert(fn == 0)


def test_collect_ner_results_both_nonempty_equal():
    utterance_targets = [[{'start': '10', 'end': '15', 'value': 'Chris',
                           'entity': 'Person'}]]
    utterance_predictions = [[{'start': '10', 'end': '15', 'value': 'Chris',
                               'entity': 'Person'}]]
    ner_filename = 'test.json'
    collect_ner_results(utterance_targets, utterance_predictions, ner_filename)
    (tp, fp, fn) = analyze_file(ner_filename)
    os.remove(ner_filename)
    assert(tp == 1)
    assert(fp == 0)
    assert(fn == 0)
