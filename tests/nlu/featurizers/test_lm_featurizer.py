import numpy as np
import pytest

from rasa.nlu.training_data import TrainingData
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.nlu.constants import TEXT, INTENT
from rasa.nlu.training_data import Message


@pytest.mark.skip(reason="Results in random crashing of github action workers")
@pytest.mark.parametrize(
    "model_name, texts, expected_shape, expected_sequence_vec, expected_cls_vec",
    [
        (
            "bert",
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (9, 768)],
            [
                [0.5727445, -0.16078179],
                [-0.5485125, 0.09632876, -0.4278888, 0.11438395, 0.18316492],
            ],
            [
                [0.068804, 0.32802248, -0.11250398, -0.11338018, -0.37116352],
                [0.05909364, 0.06433402, 0.08569086, -0.16530034, -0.11396906],
            ],
        ),
        (
            "gpt",
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
    ],
)
def test_lm_featurizer_shape_values(
    model_name, texts, expected_shape, expected_sequence_vec, expected_cls_vec
):
    transformers_config = {"model_name": model_name}

    transformers_nlp = HFTransformersNLP(transformers_config)
    lm_featurizer = LanguageModelFeaturizer()

    messages = []
    for text in texts:
        messages.append(Message.build(text=text))
    td = TrainingData(messages)

    transformers_nlp.train(td)
    lm_featurizer.train(td)

    for index in range(len(texts)):

        computed_sequence_vec, computed_sentence_vec = messages[
            index
        ].get_dense_features(TEXT, [])

        assert computed_sequence_vec.shape[0] == expected_shape[index][0] - 1
        assert computed_sequence_vec.shape[1] == expected_shape[index][1]
        assert computed_sentence_vec.shape[0] == 1
        assert computed_sentence_vec.shape[1] == expected_shape[index][1]

        # Look at the value of first dimension for a few starting timesteps
        assert np.allclose(
            computed_sequence_vec[: len(expected_sequence_vec[index]), 0],
            expected_sequence_vec[index],
            atol=1e-5,
        )

        # Look at the first value of first five dimensions
        assert np.allclose(
            computed_sentence_vec[0][:5], expected_cls_vec[index], atol=1e-5
        )

        intent_sequence_vec, intent_sentence_vec = messages[index].get_dense_features(
            INTENT, []
        )

        assert intent_sequence_vec is None
        assert intent_sentence_vec is None
