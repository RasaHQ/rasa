import numpy as np
import pytest

from rasa.nlu.training_data import TrainingData
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP
from rasa.nlu.constants import TEXT, DENSE_FEATURE_NAMES, INTENT
from rasa.nlu.training_data import Message


@pytest.mark.parametrize(
    "model_name, texts, expected_shape, expected_sequence_vec, expected_cls_vec",
    [
        (
            "bert",
            ["Good evening.", "here is the sentence I want embeddings for."],
            [(3, 768), (12, 768)],
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
            [(3, 768), (10, 768)],
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
            [(4, 768), (12, 768)],
            [
                [-0.033827826380729675, -0.10971662402153015, 0.002244209870696068],
                [
                    -0.18434514105319977,
                    -0.5386468768119812,
                    -0.11122681945562363,
                    -1.368929147720337,
                    -0.5397579669952393,
                ],
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
            [(3, 768), (11, 768)],
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
            [(3, 768), (12, 768)],
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
            [(4, 768), (12, 768)],
            [
                [-0.309267520904541, 0.12365783751010895, 0.06769893318414688],
                [
                    0.02152823843061924,
                    -0.08026768267154694,
                    -0.10808645188808441,
                    0.20090824365615845,
                    0.04756045714020729,
                ],
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

        computed_feature_vec = messages[index].get(DENSE_FEATURE_NAMES[TEXT])
        computed_sequence_vec, computed_sentence_vec = (
            computed_feature_vec[:-1],
            computed_feature_vec[-1],
        )

        assert computed_feature_vec.shape == expected_shape[index]

        # Look at the value of first dimension for a few starting timesteps
        assert np.allclose(
            computed_sequence_vec[: len(expected_sequence_vec[index]), 0],
            expected_sequence_vec[index],
            atol=1e-5,
        )

        # Look at the first value of first five dimensions
        assert np.allclose(
            computed_sentence_vec[:5], expected_cls_vec[index], atol=1e-5
        )

        intent_vec = messages[index].get(DENSE_FEATURE_NAMES[INTENT])

        assert intent_vec is None
