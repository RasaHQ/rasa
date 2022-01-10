from typing import Tuple
import tempfile
import uuid

from tqdm import tqdm 
import pandas as pd

from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

# test cases from `test_lm_featurizer_shapes_in_process_training_data`
TEST_CASES = [
        (
            "bert",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [0.6569931, 0.77279466],
                [0.21718428, 0.34955627, 0.59124136, 0.6869872, 0.16993292],
            ],
            [
                [0.29528213, 0.5543281, -0.4091331, 0.65817744, 0.81740487],
                [-0.17215663, 0.26811457, -0.1922609, -0.63926417, -1.626383],
            ],
        ),
        (
            "bert",
            "bert-base-uncased",
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [0.57274431, -0.16078192],
                [-0.54851216, 0.09632845, -0.42788929, 0.11438307, 0.18316516],
            ],
            [
                [0.06880389, 0.32802248, -0.11250392, -0.11338016, -0.37116382],
                [0.05909365, 0.06433402, 0.08569094, -0.16530040, -0.11396892],
            ],
        ),
        (
            "gpt",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [-0.0630323737859726, 0.4029877185821533],
                [
                    0.8072432279586792,
                    -0.08990508317947388,
                    0.9985930919647217,
                    -0.38779014348983765,
                    0.08921952545642853,
                ],
            ],
            [
                [
                    0.16997766494750977,
                    0.1493849903345108,
                    0.39421725273132324,
                    -0.5753618478775024,
                    0.05096133053302765,
                ],
                [
                    0.41056010127067566,
                    -0.1169343888759613,
                    -0.3019704818725586,
                    -0.40207183361053467,
                    0.6289798021316528,
                ],
            ],
        ),
        (
            "gpt2",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [-0.03382749, -0.05373593],
                [-0.18434484, -0.5386464, -0.11122551, -0.95434338, 0.28311089],
            ],
            [
                [
                    -0.04710008203983307,
                    -0.2793063223361969,
                    -0.23804056644439697,
                    -0.3212292492389679,
                    0.11430201679468155,
                ],
                [
                    -0.1809544414281845,
                    -0.017152192071080208,
                    -0.3176477551460266,
                    -0.008387327194213867,
                    0.3365338146686554,
                ],
            ],
        ),
        (
            "xlnet",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [1.7612367868423462, 2.5819129943847656],
                [
                    0.784195065498352,
                    0.7068007588386536,
                    1.5883606672286987,
                    1.891886591911316,
                    2.5209126472473145,
                ],
            ],
            [
                [
                    2.171574831008911,
                    -1.5377449989318848,
                    -3.2671749591827393,
                    0.22520869970321655,
                    -1.598855972290039,
                ],
                [
                    1.6516317129135132,
                    0.021670114248991013,
                    -2.5114030838012695,
                    1.447351098060608,
                    -2.5866634845733643,
                ],
            ],
        ),
        (
            "distilbert",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [0.22866562008857727, -0.0575055330991745],
                [
                    -0.6448041796684265,
                    -0.5105321407318115,
                    -0.4892978072166443,
                    0.17531153559684753,
                    0.22717803716659546,
                ],
            ],
            [
                [
                    -0.09814466536045074,
                    -0.07325993478298187,
                    0.22358475625514984,
                    -0.20274735987186432,
                    -0.07363069802522659,
                ],
                [
                    -0.146609365940094,
                    -0.07373693585395813,
                    0.016850866377353668,
                    -0.2407529354095459,
                    -0.0979844480752945,
                ],
            ],
        ),
        (
            "roberta",
            None,
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [-0.3092685, 0.09567838],
                [0.02152853, -0.08026707, -0.1080862, 0.12423468, -0.05378958],
            ],
            [
                [
                    -0.03930358216166496,
                    0.034788478165864944,
                    0.12246038764715195,
                    0.08401528000831604,
                    0.7026961445808411,
                ],
                [
                    -0.018586941063404083,
                    -0.09835464507341385,
                    0.03242188319563866,
                    0.09366855770349503,
                    0.4458026587963104,
                ],
            ],
        ),
    ]

_parameters = "model_name, model_weights, texts, expected_shape, expected_sequence_vec, expected_cls_vec"
TEST_CASES = [ {key : item for key, item in zip(_parameters.split(', '), test_case)} for test_case in TEST_CASES]

def get_featurizer(test_case) -> Tuple[str, LanguageModelFeaturizer]:
    config = {"model_name": test_case['model_name'], "model_weights" : test_case['model_weights']}
    return LanguageModelFeaturizer(
        config = {**LanguageModelFeaturizer.get_default_config(), **config},
        execution_context= ExecutionContext(GraphSchema({}), uuid.uuid4().hex),
    )

def run(test_case) -> pd.DataFrame:
    # simluates only `test_lm_featurizer_shapes_in_process_training_data`
    # but `test_lm_featurizer_shapes_in_process_messages` should do the same
    # thing
    
    lm_featurizer = get_featurizer(test_case)
    
    whitespace_tokenizer = WhitespaceTokenizer(WhitespaceTokenizer.get_default_config())
    
    messages = [Message.build(text=text) for text in test_case['texts']]
    td = TrainingData(messages)
    whitespace_tokenizer.process_training_data(td)
    td = lm_featurizer.process_training_data(td)

    comparison = pd.DataFrame()
    for index in range(len(messages)): 
        expected_shape = test_case['expected_shape']
        expected_sequence_vec = test_case['expected_sequence_vec']
        expected_cls_vec = test_case['expected_cls_vec']

        (computed_sequence_vec, computed_sentence_vec) = messages[
            index
        ].get_dense_features('text', [])
        if computed_sequence_vec:
            computed_sequence_vec = computed_sequence_vec.features
        if computed_sentence_vec:
            computed_sentence_vec = computed_sentence_vec.features


        assert computed_sequence_vec.shape[0] == expected_shape[index][0] - 1
        assert computed_sequence_vec.shape[1] == expected_shape[index][1]
        assert computed_sentence_vec.shape[0] == 1
        assert computed_sentence_vec.shape[1] == expected_shape[index][1]

        (intent_sequence_vec, intent_sentence_vec) = messages[
            index
        ].get_dense_features('intent', [])
        if intent_sequence_vec:
            intent_sequence_vec = intent_sequence_vec.features
        if intent_sentence_vec:
            intent_sentence_vec = intent_sentence_vec.features

        assert intent_sequence_vec is None
        assert intent_sentence_vec is None

        def max_abs_diff(a,b):
            return max([abs(x-y) for x,y in zip(a,b)])
        row =  {
            'model_name' : test_case['model_name'],
            'model_weights' : test_case['model_weights'],
            'text' : messages[index].get('text'),
            'sequence_expected' : computed_sequence_vec[: len(expected_sequence_vec[index]), 0],
            'sequence_actual' : expected_sequence_vec[index],
            'sentence_expected' : expected_cls_vec[index],
            'sentence_actual' : computed_sentence_vec[0][:5],
        }
        for key in ['sequence', 'sentence']:
            row[f"{key}_diff"] = max_abs_diff(row[f"{key}_expected"], row[f"{key}_actual"])
        comparison = comparison.append(row, ignore_index=True)
        
    comparison = comparison.reset_index(drop=True)
    return comparison

def collect_comparisons(test_cases) -> pd.DataFrame:
    comparisons = pd.DataFrame()
    for test_case in tqdm(TEST_CASES):
        comparison = run(test_case)
        comparisons = comparisons.append(comparison, ignore_index=True)
    comparisons['max_diff'] = comparisons[['sequence_diff', 'sentence_diff']].max(axis=1)
    return comparisons.reset_index(drop=True)