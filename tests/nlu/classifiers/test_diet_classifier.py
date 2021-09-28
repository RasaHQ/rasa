import copy
from pathlib import Path
import numpy as np
import pytest
import random
from unittest.mock import Mock
from typing import List, Text, Dict, Any, Optional, Tuple
from _pytest.monkeypatch import MonkeyPatch
import itertools
import re

import rasa.model
from rasa.shared.exceptions import InvalidConfigException
import rasa.nlu.train
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_VALUE,
    TEXT,
    INTENT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.constants import (
    FEATURIZERS,
    LABEL,
    LOSS_TYPE,
    MASK,
    RANDOM_SEED,
    RANKING_LENGTH,
    EPOCHS,
    MASKED_LM,
    SENTENCE,
    SEQUENCE,
    SEQUENCE_LENGTH,
    TENSORBOARD_LOG_LEVEL,
    TENSORBOARD_LOG_DIR,
    EVAL_NUM_EPOCHS,
    EVAL_NUM_EXAMPLES,
    CONSTRAIN_SIMILARITIES,
    CHECKPOINT_MODEL,
    BILOU_FLAG,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    MODEL_CONFIDENCE,
    LINEAR_NORM,
)
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.constants import (
    BILOU_ENTITIES,
    TOKENS_NAMES,
)
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.classifiers.diet_classifier import (
    LABEL_KEY,
    LABEL_SUB_KEY,
    DIETClassifier,
)
from rasa.nlu.model import Interpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils import train_utils
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data.loading import load_data
from rasa.utils.tensorflow.model_data_utils import FeatureArray
from rasa.utils.tensorflow.exceptions import RasaModelConfigException
from tests.nlu.featurizers.dummy_feature_generation import (
    FeaturizerDescription,
    FeatureGenerator,
)


def test_init_raises_when_there_is_no_target():
    with pytest.raises(RasaModelConfigException, match="Model is neither asked to"):
        DIETClassifier(
            component_config={INTENT_CLASSIFICATION: False, ENTITY_RECOGNITION: False,}
        )


class SimpleIntentClassificationTestCaseWithEntities(FeatureGenerator):
    """Generates a simple intent classification test case with entities.

    The data includes:
    - 'NLU' messages with text, intents and possibly entities
    - 'Core' messages with either text or intent (and entities)

    The test case data does **not** include:
    - empty messages which would be classified as 'NLU' messages
    - messages containing action names or texts which would be classified as
      core messages

    For a full docstring see parent class.
    """

    def generate_input_and_concatenated_features(
        self, input_contains_features_for_intent: bool,
    ) -> Tuple[List[Message], Dict[Text, Any]]:
        """Creates input messages and concatentations of their features.

        Creates a list of dummy messages with texts, tokens, intents, and entities
        that contain some randomly created dense features, one for each of the given
        featurizer names given in the `featurizers` list.
        For each attribute, the function separately creates matrices that represent
        a concatenation of the dense features created by the `used_featurizers`.

        Args:
            input_contains_features_for_intent: whether the input should contain
                features for the `INTENT` attribute

        Returns:
            the messages and a nested mapping from feature type and attribute to
                2d-matrices containing the features generated for the chosen
                `used_featurizers` and all messages
        """
        # prepare data for one of the more complex messages first
        text = "the city of bielefeld does not exist"
        tokens = [
            Token(text=match.group(), start=match.start())
            for match in re.finditer(r"\w+", text)
        ]
        entities = [
            {
                ENTITY_ATTRIBUTE_VALUE: "city of bielefeld",
                ENTITY_ATTRIBUTE_START: tokens[1].start,
                ENTITY_ATTRIBUTE_END: tokens[3].end,
                ENTITY_ATTRIBUTE_TYPE: "city",
            },
            {
                ENTITY_ATTRIBUTE_VALUE: tokens[-1].text,
                ENTITY_ATTRIBUTE_START: tokens[-1].start,
                ENTITY_ATTRIBUTE_END: tokens[-1].end,
                ENTITY_ATTRIBUTE_TYPE: "what",
            },
        ]

        # create messages
        messages = [
            # message 0: 'intent1' (nlu message)
            Message(
                data={
                    TEXT: "bla",
                    INTENT: "intent1",
                    TOKENS_NAMES[TEXT]: [Token(text="bla", start=0)],
                    "other": "unrelated-attribute1",
                },
                ENTITIES=[],
            ),
            # message 1: 'intent2' (nlu message)
            Message(
                data={
                    TEXT: "blub",
                    INTENT: "intent2",
                    TOKENS_NAMES[TEXT]: [Token(text="blub", start=0)],
                    "other": "unrelated-attribute2",
                },
                ENTITIES=[],
            ),
            # message 2: 'intent2' + 2 entities ('city','what') (nlu message)
            Message(
                data={
                    TEXT: text,
                    INTENT: "intent2",
                    TOKENS_NAMES[TEXT]: tokens,
                    "other": "unrelated-attribute3",
                    ENTITIES: entities,
                }
            ),
            # message 3: like message 2 but *no* intent (core message)
            Message(
                data={
                    TEXT: text,
                    TOKENS_NAMES[TEXT]: tokens,
                    "other": "unrelated-attribute3",
                    ENTITIES: entities,
                }
            ),
            # message 4: like message 2 but *no* text  (core message)
            Message(
                data={
                    TEXT: "",  # an empty text is like no text
                    INTENT: "intent2",
                    TOKENS_NAMES[TEXT]: tokens,
                    "other": "unrelated-attribute3",
                    ENTITIES: entities,
                }
            ),
        ]

        # we expect feature to be extracted for the following features
        attributes_with_features = [TEXT]
        if input_contains_features_for_intent:
            attributes_with_features.append(INTENT)
        concatenated_features = self.generate_features_for_all_messages(
            messages, add_to_messages=True, attributes=attributes_with_features,
        )

        # Add features for some unrelated attributes to the messages
        # (to illustrate that this attribute will not show up in the final model data)
        _ = self.generate_features_for_all_messages(
            messages, add_to_messages=True, attributes=["other"],
        )

        return messages, concatenated_features


@pytest.mark.parametrize(
    (
        "intent_classification, entity_recognition, bilou_tagging, "
        "input_contains_features_for_intent, used_featurizers"
    ),
    [
        (
            intent_classification,
            entity_recognition,
            bilou_tagging,
            input_contains_features_for_intent,
            used_featurizers,
        )
        for (
            intent_classification,
            entity_recognition,
            bilou_tagging,
            input_contains_features_for_intent,
        ) in itertools.product([True, False], repeat=4)
        for used_featurizers in [None, ["2", "1"]]
        if (intent_classification or entity_recognition)
        and not (bilou_tagging and not entity_recognition)
    ],
)
def test_preprocess_train_data_and_create_data_for_prediction(
    intent_classification: bool,
    entity_recognition: bool,
    bilou_tagging: bool,
    input_contains_features_for_intent: bool,
    used_featurizers: Optional[List[Text]],
):

    # Create and choose featurizers
    featurizer_descriptions = [
        FeaturizerDescription(
            name="1", seq_feat_dim=2, sent_feat_dim=2, sparse=True, dense=False
        ),
        FeaturizerDescription(
            name="2", seq_feat_dim=4, sent_feat_dim=4, sparse=False, dense=False
        ),
        FeaturizerDescription(
            name="3", seq_feat_dim=8, sent_feat_dim=8, sparse=True, dense=True
        ),
    ]
    used_featurizers = used_featurizers or ["1", "2", "3"]
    assert set(used_featurizers) <= {"1", "2", "3"}

    # Create the model
    model = DIETClassifier(
        component_config={
            INTENT_CLASSIFICATION: intent_classification,
            ENTITY_RECOGNITION: entity_recognition,
            FEATURIZERS: used_featurizers,
            BILOU_FLAG: bilou_tagging,
            EPOCHS: 1,
            CONSTRAIN_SIMILARITIES: True,
        }
    )

    # Create input and expected output
    test_case = SimpleIntentClassificationTestCaseWithEntities(
        featurizer_descriptions, used_featurizers=used_featurizers
    )
    messages, expected_outputs = test_case.generate_input_and_concatenated_features(
        input_contains_features_for_intent=input_contains_features_for_intent,
    )

    # Preprocess Training Data
    messages_for_training = copy.deepcopy(messages)
    model_data_for_training = model.preprocess_train_data(
        training_data=TrainingData(messages_for_training)
    )

    # Check index to label id mapping
    if intent_classification:
        # Note that this mapping is needed by predict, so we test this separately.
        assert model.index_label_id_mapping == {0: "intent1", 1: "intent2"}
    else:
        # we only create a dummy mapping in this case since RasaModel can't deal
        # with empty label data
        assert model.index_label_id_mapping == {0: "0"}

    # Imitate Creation of Model Data during Predict
    messages_for_prediction = copy.deepcopy(messages)
    model_data_for_prediction = model._create_model_data(
        messages=messages_for_prediction,
        training=False,
        label_id_dict=model.index_label_id_mapping,
    )

    # Compare with expected results

    for data, training, only_label_data in [
        (model_data_for_training, True, False),
        (model._label_data, True, True),
        (model_data_for_prediction, False, False),
    ]:
        #
        if only_label_data and not intent_classification:
            # if we're not predicted an intent, then the label data will not be used
            continue

        # Which messages do we need to look at?
        if training:
            if not only_label_data:
                # only the first three ("nlu") messages will be used for training
                relevant_indices = [0, 1, 2]
                expected_mhot_label_features = [[[1, 0]], [[0, 1]], [[0, 1]]]
                expected_id_label_features = [0, 1, 1]

            else:
                # only one example per label in the right oder is needed
                relevant_indices = [0, 1]
                expected_mhot_label_features = [[[1, 0]], [[0, 1]]]
                expected_id_label_features = [0, 1]
        else:
            relevant_indices = range(len(messages))

        # Check size and attributes
        assert data.number_of_examples() == len(relevant_indices)
        expected_keys = set()
        if not only_label_data:
            expected_keys.add(TEXT)
        if training and intent_classification:
            expected_keys.add(LABEL)
        if training and entity_recognition and not only_label_data:
            expected_keys.add(ENTITIES)
        assert set(data.keys()) == expected_keys

        # TODO: test that DIET is able to fill in sentence features where
        # required - if that is needed/used at all

        if not only_label_data:
            # NOTE: Observe that DIET always uses sequence features for text input
            # (no matter the configuration) which is why we do not need to test whether
            # the data perpartion skips sequence features.
            # This does happens and get tested for the `ResponseSelector`.

            # Check subkeys for TEXT
            expected_text_sub_keys = {MASK, SENTENCE, SEQUENCE, SEQUENCE_LENGTH}
            assert set(data.get(TEXT).keys()) == expected_text_sub_keys
            text_features = data.get(TEXT)
            # - subkey: mask (this is a "turn" mask)
            # TODO: Masks are irrelevant for DIET. Check whether we can remove them.
            # Note: mask entry will be 0 if text is None but 1 if text is ""
            mask_features = text_features.get(MASK)
            assert len(mask_features) == 1
            mask = np.array(mask_features[0])  # because it's a feature array
            assert mask.shape == (len(relevant_indices), 1, 1)
            assert np.all(mask == 1)
            # - subkey: sequence-length
            length_features = text_features.get(SEQUENCE_LENGTH)
            assert len(length_features) == 1
            lengths = np.array(length_features[0])  # because it's a feature array
            assert lengths.shape == (len(relevant_indices),)
            if not training:
                # the last message contains no text, just an intent
                assert np.all(lengths[:-1] == test_case.seq_len)
                assert lengths[-1] == 0
            else:
                # in this case we only use "nlu" messages - which have a text
                assert np.all(lengths == test_case.seq_len)
            # - subkey: sequence / sentence
            test_case.compare_features_of_same_type_and_sparseness(
                actual=text_features,
                expected=expected_outputs[TEXT],
                indices_of_expected=relevant_indices,
            )

            # TODO: check sparse feature sizes

        if training and intent_classification:

            # Check subkeys for LABEL
            expected_label_subkeys = {LABEL_SUB_KEY, SENTENCE}
            if not only_label_data:
                expected_label_subkeys.add(MASK)
            if input_contains_features_for_intent:
                expected_label_subkeys.update({SEQUENCE, SEQUENCE_LENGTH})
            label_features = data.get(LABEL_KEY)
            assert set(label_features.keys()) == expected_label_subkeys
            # - subkey: mask (this is a "turn" mask, hence all masks are just "[1]")
            if not only_label_data:
                assert len(label_features.get(MASK)) == 1
                mask = np.array(label_features.get(MASK)[0])
                assert np.all(mask == [1] * len(relevant_indices))
            # - subkey: label_sub_key / id
            id_features = label_features.get(LABEL_SUB_KEY)
            assert len(id_features) == 1
            assert id_features[0].shape == (len(relevant_indices), 1)
            ids = np.array(id_features[0].flatten())
            assert np.all(ids == expected_id_label_features)
            # - subkey: sentence
            if not input_contains_features_for_intent:
                # - subkey: sequence and sequence length
                assert SEQUENCE not in label_features
                assert SEQUENCE_LENGTH not in label_features
                # - subkey: sentence
                assert len(label_features.get(SENTENCE)) == 1
                generated_label_features = np.array(label_features.get(SENTENCE)[0])
                assert np.all(generated_label_features == expected_mhot_label_features)
            else:
                # - subkey: sequence-length
                length_features = label_features.get(SEQUENCE_LENGTH)
                assert len(length_features) == 1
                lengths = np.array(length_features[0])
                assert lengths.shape == (len(relevant_indices),)
                assert np.all(lengths == test_case.seq_len)
                # - subkeys: sequence and sentence
                test_case.compare_features_of_same_type_and_sparseness(
                    expected=expected_outputs[INTENT],
                    actual=label_features,
                    indices_of_expected=relevant_indices,
                )

        else:
            assert not data.get(LABEL_KEY)

        if training and entity_recognition and not only_label_data:

            # Check sub-keys for ENTITIES
            expected_entity_sub_keys = {ENTITY_ATTRIBUTE_TYPE, MASK}
            entity_features = data.get(ENTITIES)
            assert entity_features
            assert set(entity_features.keys()) == expected_entity_sub_keys
            # - every subkey is a list containing one feature array
            for key in entity_features:
                assert len(entity_features.get(key)) == 1
                assert isinstance(entity_features.get(key)[0], FeatureArray)
            # - subkey: mask  (this is a "turn" mask, hence all masks are just "[1]")
            mask = np.array(entity_features.get(MASK)[0])
            expected_mask = [1] * len(relevant_indices)
            assert np.all(mask == expected_mask)
            # - subkey: entities
            entity_sub_features = entity_features[ENTITY_ATTRIBUTE_TYPE][0]
            assert len(entity_sub_features) == len(relevant_indices)
            # message 0+1: no entity
            for idx in [0, 1]:
                assert np.all(np.array(entity_sub_features[idx]).flatten() == [0])
            # message 2: has entity1 and entity2
            for idx in [2]:
                if not bilou_tagging:
                    expected_entity_features = [0, 1, 1, 1, 0, 0, 2]
                    # where 0 = no entity, 1 = city, 2 = what
                else:
                    message = messages_for_training[idx]
                    assert message.get(BILOU_ENTITIES) == [
                        "O",
                        "B-city",
                        "I-city",
                        "L-city",
                        "O",
                        "O",
                        "U-what",
                    ]
                    expected_entity_features = [0, 1, 2, 3, 0, 0, 8]
                    # where 0 = no entity, 1/2/3/4 = B/I/L/U-city, and
                    # 5/6/7/8 = B/I/L/U-what
                assert np.all(
                    np.array(entity_sub_features[idx]).flatten()
                    == expected_entity_features
                )
            # message 3-4: are not considered during training
            assert not {3.4}.intersection(relevant_indices)
        else:
            assert not data.get(ENTITIES)


@pytest.mark.parametrize(
    "should_raise", [True, False],
)
def test_preprocess_training_data_raises_if_sentence_features_are_missing(
    should_raise: bool,
):
    classifier = DIETClassifier({"intent_classification": True})
    sequence_feature = Features(
        features=np.zeros((2, 1)), attribute=TEXT, feature_type=SEQUENCE, origin=""
    )
    features = [sequence_feature]
    if not should_raise:
        sentence_feature = Features(
            features=np.zeros((1, 1)), attribute=TEXT, feature_type=SENTENCE, origin=""
        )
        features.append(sentence_feature)
    message = Message(
        data={TEXT: "dummy_text", INTENT: "dummy_intent"}, features=features
    )
    training_data = TrainingData([message])

    if should_raise:
        message = "Expected all featurizers to produce sentence features."
        with pytest.raises(RasaModelConfigException, match=message):
            classifier.preprocess_train_data(training_data)
    else:
        classifier.preprocess_train_data(training_data)


@pytest.mark.parametrize("num_messages_per_label", ([1, 4], [4, 0], [0, 4]))
def test_collect_one_example_per_label(num_messages_per_label: List[int]):

    # make sure messages, labels and ids are unique and not as simple as 0,1,2,...
    def idx2labelstr(idx: int) -> Text:
        return f"{idx**2}"

    def idx2labelid(idx: int) -> int:
        # Note: this must be an increasing function otherwise the tests below will
        # fail as the tested function sorts
        return idx * 2 + 1

    messages_per_label = {
        idx: [
            Message(data={"label": idx2labelstr(idx), "text": f"{msg_idx}"})
            for msg_idx in range(num_messages)
        ]
        for idx, num_messages in enumerate(num_messages_per_label)
    }
    label_id_dict = {
        idx2labelstr(idx): idx2labelid(idx)
        for idx in range(len(num_messages_per_label))
    }

    # collect all messages and shuffle
    messages = [msg for msg_lst in messages_per_label.values() for msg in msg_lst]
    random.shuffle(messages)

    # apply and check
    if any(num == 0 for num in num_messages_per_label):
        with pytest.raises(
            InvalidConfigException, match="Expected at least one example for each label"
        ):
            DIETClassifier._collect_one_example_per_label(
                messages=messages, label_id_dict=label_id_dict, label_attribute="label"
            )
    else:
        (
            sorted_labelidx,
            sorted_messages,
        ) = DIETClassifier._collect_one_example_per_label(
            messages=messages, label_id_dict=label_id_dict, label_attribute="label"
        )
        # Because of the way labelids are created from the indices, we have...
        assert all(
            [
                returned_id == idx2labelid(idx)
                for idx, returned_id in zip(
                    range(len(num_messages_per_label)), sorted_labelidx
                )
            ]
        )
        for idx in range(len(sorted_labelidx)):
            assert sorted_messages[idx].data["label"] == idx2labelstr(idx)


def test_compute_default_label_features():
    num_labels = 4
    output = DIETClassifier._compute_default_label_features(num_labels=num_labels)

    assert len(output) == 1
    output = output[0]

    for i, o in enumerate(output):
        assert isinstance(o, np.ndarray)
        assert o.shape == (1, num_labels)
        assert o[0][i] == 1


def get_checkpoint_dir_path(path: Path, model_dir: Path) -> Path:
    """
    Produce the path of the checkpoint directory for DIET.

    This is coupled to the persist method of DIET.

    Args:
        model_dir: the model directory
        path: the path passed to train for training output.

    """
    return path / model_dir / "checkpoints"


@pytest.mark.parametrize(
    "messages, entity_expected",
    [
        (
            [
                Message(
                    data={
                        TEXT: "test a",
                        INTENT: "intent a",
                        ENTITIES: [
                            {"start": 0, "end": 4, "value": "test", "entity": "test"}
                        ],
                    },
                ),
                Message(
                    data={
                        TEXT: "test b",
                        INTENT: "intent b",
                        ENTITIES: [
                            {"start": 0, "end": 4, "value": "test", "entity": "test"}
                        ],
                    },
                ),
            ],
            True,
        ),
        (
            [
                Message(data={TEXT: "test a", INTENT: "intent a"},),
                Message(data={TEXT: "test b", INTENT: "intent b"},),
            ],
            False,
        ),
    ],
)
def test_model_data_signature_with_entities(
    messages: List[Message], entity_expected: bool
):
    classifier = DIETClassifier({"BILOU_flag": False})

    training_data = TrainingData(messages)
    FeatureGenerator().generate_features_for_all_messages(
        training_data.intent_examples, attributes=[TEXT], add_to_messages=True,
    )

    # create tokens for entity parsing inside DIET
    tokenizer = WhitespaceTokenizer()
    tokenizer.train(training_data)

    model_data = classifier.preprocess_train_data(training_data)
    entity_exists = "entities" in model_data.get_signature().keys()
    assert entity_exists == entity_expected


async def _train_persist_load_with_different_settings(
    pipeline: List[Dict[Text, Any]],
    component_builder: ComponentBuilder,
    tmp_path: Path,
    should_finetune: bool,
):
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data="data/examples/rasa/demo-rasa-multi-intent.yml",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(
        persisted_path,
        component_builder,
        new_config=_config if should_finetune else None,
    )

    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") == trained.parse("Rasa is great!")


@pytest.mark.skip_on_windows
@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_different_settings_non_windows(
    component_builder: ComponentBuilder, tmp_path: Path
):
    pipeline = [
        {
            "name": "WhitespaceTokenizer",
            "intent_tokenization_flag": True,
            "intent_split_symbol": "+",
        },
        {"name": "CountVectorsFeaturizer"},
        {"name": "DIETClassifier", MASKED_LM: True, EPOCHS: 1},
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmp_path, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmp_path, should_finetune=True
    )


@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_different_settings(component_builder, tmpdir):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {"name": "DIETClassifier", LOSS_TYPE: "margin", EPOCHS: 1},
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=True
    )


@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_only_entity_recognition(
    component_builder, tmpdir
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "DIETClassifier",
            ENTITY_RECOGNITION: True,
            INTENT_CLASSIFICATION: False,
            EPOCHS: 1,
        },
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=True
    )


@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_only_intent_classification(
    component_builder, tmpdir
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "DIETClassifier",
            ENTITY_RECOGNITION: False,
            INTENT_CLASSIFICATION: True,
            EPOCHS: 1,
        },
    ]
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=False
    )
    await _train_persist_load_with_different_settings(
        pipeline, component_builder, tmpdir, should_finetune=True
    )


async def test_raise_error_on_incorrect_pipeline(
    component_builder, tmp_path: Path, nlu_as_json_path: Text
):
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "DIETClassifier", EPOCHS: 1},
            ],
            "language": "en",
        }
    )

    with pytest.raises(Exception) as e:
        await rasa.nlu.train.train(
            _config,
            path=str(tmp_path),
            data=nlu_as_json_path,
            component_builder=component_builder,
        )

    assert "'DIETClassifier' requires 'Featurizer'" in str(e.value)


def as_pipeline(*components):
    return [{"name": c} for c in components]


@pytest.mark.parametrize(
    "classifier_params, data_path, output_length, output_should_sum_to_1",
    [
        (
            {RANDOM_SEED: 42, EPOCHS: 1},
            "data/test/many_intents.yml",
            10,
            True,
        ),  # default config
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 0, EPOCHS: 1},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            False,
        ),  # no normalization
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 3, EPOCHS: 1},
            "data/test/many_intents.yml",
            3,
            True,
        ),  # lower than default ranking_length
        (
            {RANDOM_SEED: 42, RANKING_LENGTH: 12, EPOCHS: 1},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            False,
        ),  # higher than default ranking_length
        (
            {RANDOM_SEED: 42, EPOCHS: 1},
            "data/test_moodbot/data/nlu.yml",
            7,
            True,
        ),  # less intents than default ranking_length
    ],
)
async def test_softmax_normalization(
    component_builder,
    tmp_path,
    classifier_params,
    data_path: Text,
    output_length,
    output_should_sum_to_1,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=data_path,
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")
    # check that the output was correctly truncated after normalization
    assert len(intent_ranking) == output_length

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in intent_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1 == output_should_sum_to_1

    # check whether the normalization of rankings is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.parametrize(
    "classifier_params, data_path",
    [
        (
            {
                RANDOM_SEED: 42,
                EPOCHS: 1,
                MODEL_CONFIDENCE: LINEAR_NORM,
                RANKING_LENGTH: -1,
            },
            "data/test_moodbot/data/nlu.yml",
        ),
    ],
)
async def test_inner_linear_normalization(
    component_builder: ComponentBuilder,
    tmp_path: Path,
    classifier_params: Dict[Text, Any],
    data_path: Text,
    monkeypatch: MonkeyPatch,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=data_path,
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in intent_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1

    # check whether the normalization of rankings is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]

    # normalize shouldn't have been called
    mock.normalize.assert_not_called()


@pytest.mark.parametrize(
    "classifier_params, output_length",
    [({LOSS_TYPE: "margin", RANDOM_SEED: 42, EPOCHS: 1}, LABEL_RANKING_LENGTH)],
)
async def test_margin_loss_is_not_normalized(
    monkeypatch, component_builder, tmpdir, classifier_params, output_length
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data="data/test/many_intents.yml",
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")

    # check that the output was not normalized
    mock.normalize.assert_not_called()

    # check that the output was correctly truncated
    assert len(intent_ranking) == output_length

    # make sure top ranking is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.timeout(120, func_only=True)
async def test_set_random_seed(component_builder, tmpdir, nlu_as_json_path: Text):
    """test if train result is the same for two runs of tf embedding"""

    # set fixed random seed
    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "DIETClassifier", RANDOM_SEED: 1, EPOCHS: 1},
            ],
            "language": "en",
        }
    )

    # first run
    (trained_a, _, persisted_path_a) = await rasa.nlu.train.train(
        _config,
        path=tmpdir.strpath + "_a",
        data=nlu_as_json_path,
        component_builder=component_builder,
    )
    # second run
    (trained_b, _, persisted_path_b) = await rasa.nlu.train.train(
        _config,
        path=tmpdir.strpath + "_b",
        data=nlu_as_json_path,
        component_builder=component_builder,
    )

    loaded_a = Interpreter.load(persisted_path_a, component_builder)
    loaded_b = Interpreter.load(persisted_path_b, component_builder)
    result_a = loaded_a.parse("hello")["intent"]["confidence"]
    result_b = loaded_b.parse("hello")["intent"]["confidence"]

    assert result_a == result_b


@pytest.mark.parametrize("log_level", ["epoch", "batch"])
async def test_train_tensorboard_logging(
    log_level: Text,
    component_builder: ComponentBuilder,
    tmpdir: Path,
    nlu_data_path: Text,
):
    tensorboard_log_dir = Path(tmpdir / "tensorboard")

    assert not tensorboard_log_dir.exists()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {
                    "name": "CountVectorsFeaturizer",
                    "analyzer": "char_wb",
                    "min_ngram": 3,
                    "max_ngram": 17,
                    "max_features": 10,
                    "min_df": 5,
                },
                {
                    "name": "DIETClassifier",
                    EPOCHS: 1,
                    TENSORBOARD_LOG_LEVEL: log_level,
                    TENSORBOARD_LOG_DIR: str(tensorboard_log_dir),
                    MODEL_CONFIDENCE: "linear_norm",
                    CONSTRAIN_SIMILARITIES: True,
                    EVAL_NUM_EXAMPLES: 15,
                    EVAL_NUM_EPOCHS: 1,
                },
            ],
            "language": "en",
        }
    )

    await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data=nlu_data_path,
        component_builder=component_builder,
    )

    assert tensorboard_log_dir.exists()

    all_files = list(tensorboard_log_dir.rglob("*.*"))
    assert len(all_files) == 2


async def test_train_model_checkpointing(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-checkpointed-model"
    model_dir = tmp_path / model_name
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, model_dir)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 2,
                    EVAL_NUM_EPOCHS: 1,
                    EVAL_NUM_EXAMPLES: 10,
                    CHECKPOINT_MODEL: True,
                },
            ],
            "language": "en",
        }
    )

    await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=nlu_data_path,
        component_builder=component_builder,
        fixed_model_name=model_name,
    )

    assert checkpoint_dir.is_dir()

    """
    Tricky to validate the *exact* number of files that should be there, however there
    must be at least the following:
        - metadata.json
        - checkpoint
        - component_1_CountVectorsFeaturizer (as per the pipeline above)
        - component_2_DIETClassifier files (more than 1 file)
    """
    all_files = list(model_dir.rglob("*.*"))
    assert len(all_files) > 4


async def test_train_model_not_checkpointing(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-not-checkpointed-model"
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, tmp_path / model_name)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {"name": "DIETClassifier", EPOCHS: 2, CHECKPOINT_MODEL: False},
            ],
            "language": "en",
        }
    )

    await rasa.nlu.train.train(
        _config,
        path=str(tmp_path),
        data=nlu_data_path,
        component_builder=component_builder,
        fixed_model_name=model_name,
    )

    assert not checkpoint_dir.is_dir()


async def test_train_fails_with_zero_eval_num_epochs(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-fail"
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, tmp_path / model_name)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 1,
                    CHECKPOINT_MODEL: True,
                    EVAL_NUM_EPOCHS: 0,
                    EVAL_NUM_EXAMPLES: 10,
                },
            ],
            "language": "en",
        }
    )
    with pytest.raises(InvalidConfigException):
        with pytest.warns(UserWarning) as warning:
            await rasa.nlu.train.train(
                _config,
                path=str(tmp_path),
                data=nlu_data_path,
                component_builder=component_builder,
                fixed_model_name=model_name,
            )
    assert not checkpoint_dir.is_dir()
    warn_text = (
        f"You have opted to save the best model, but the value of '{EVAL_NUM_EPOCHS}' "
        f"is not -1 or greater than 0. Training will fail."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1


async def test_doesnt_checkpoint_with_zero_eval_num_examples(
    component_builder: ComponentBuilder, tmp_path: Path, nlu_data_path: Text,
):
    model_name = "nlu-fail-checkpoint"
    checkpoint_dir = get_checkpoint_dir_path(tmp_path, tmp_path / model_name)
    assert not checkpoint_dir.is_dir()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "DIETClassifier",
                    EPOCHS: 2,
                    CHECKPOINT_MODEL: True,
                    EVAL_NUM_EXAMPLES: 0,
                    EVAL_NUM_EPOCHS: 1,
                },
            ],
            "language": "en",
        }
    )
    with pytest.warns(UserWarning) as warning:
        await rasa.nlu.train.train(
            _config,
            path=str(tmp_path),
            data=nlu_data_path,
            component_builder=component_builder,
            fixed_model_name=model_name,
        )

    assert not checkpoint_dir.is_dir()
    warn_text = (
        f"You have opted to save the best model, but the value of "
        f"'{EVAL_NUM_EXAMPLES}' is not greater than 0. No checkpoint model "
        f"will be saved."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1


@pytest.mark.parametrize(
    "classifier_params",
    [
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: False},
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: True},
    ],
)
@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_composite_entities(
    classifier_params, component_builder, tmpdir
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "DIETClassifier"
    )
    assert pipeline[2]["name"] == "DIETClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trainer, trained, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=tmpdir.strpath,
        data="data/test/demo-rasa-composite-entities.yml",
        component_builder=component_builder,
    )

    assert trainer.pipeline
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)

    assert loaded.pipeline
    text = "I am looking for an italian restaurant"
    assert loaded.parse(text) == trained.parse(text)


async def test_process_gives_diagnostic_data(
    response_selector_interpreter: Interpreter,
):
    """Tests if processing a message returns attention weights as numpy array."""
    interpreter = response_selector_interpreter
    message = Message(data={TEXT: "hello"})
    for component in interpreter.pipeline:
        component.process(message)

    diagnostic_data = message.get(DIAGNOSTIC_DATA)

    # The last component is DIETClassifier, which should add attention weights
    name = f"component_{len(interpreter.pipeline) - 2}_DIETClassifier"
    assert isinstance(diagnostic_data, dict)
    assert name in diagnostic_data
    assert "attention_weights" in diagnostic_data[name]
    assert isinstance(diagnostic_data[name].get("attention_weights"), np.ndarray)
    assert "text_transformed" in diagnostic_data[name]
    assert isinstance(diagnostic_data[name].get("text_transformed"), np.ndarray)


@pytest.mark.timeout(120)
async def test_adjusting_layers_incremental_training(
    component_builder: ComponentBuilder, tmpdir: Path
):
    """Tests adjusting sparse layers of `DIETClassifier` to increased sparse
       feature sizes during incremental training.

       Testing is done by checking the layer sizes.
       Checking if they were replaced correctly is also important
       and is done in `test_replace_dense_for_sparse_layers`
       in `test_rasa_layers.py`.
    """
    iter1_data_path = "data/test_incremental_training/iter1/"
    iter2_data_path = "data/test_incremental_training/"
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "LexicalSyntacticFeaturizer"},
        {"name": "RegexFeaturizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "CountVectorsFeaturizer",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        {"name": "DIETClassifier", EPOCHS: 1},
    ]
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (_, trained, persisted_path) = await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data=iter1_data_path,
        component_builder=component_builder,
    )
    assert trained.pipeline
    old_data_signature = trained.pipeline[-1].model.data_signature
    old_predict_data_signature = trained.pipeline[-1].model.predict_data_signature
    message = Message.build(text="Rasa is great!")
    trained.featurize_message(message)
    old_sparse_feature_sizes = message.get_sparse_feature_sizes(attribute=TEXT)
    initial_diet_layers = (
        trained.pipeline[-1]
        .model._tf_layers["sequence_layer.text"]
        ._tf_layers["feature_combining"]
    )
    initial_diet_sequence_layer = initial_diet_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    initial_diet_sentence_layer = initial_diet_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    initial_diet_sequence_size = initial_diet_sequence_layer.get_kernel().shape[0]
    initial_diet_sentence_size = initial_diet_sentence_layer.get_kernel().shape[0]
    assert initial_diet_sequence_size == sum(
        old_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert initial_diet_sentence_size == sum(
        old_sparse_feature_sizes[FEATURE_TYPE_SENTENCE]
    )

    loaded = Interpreter.load(persisted_path, component_builder, new_config=_config,)
    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") == trained.parse("Rasa is great!")
    (_, trained, _) = await rasa.nlu.train.train(
        _config,
        path=str(tmpdir),
        data=iter2_data_path,
        component_builder=component_builder,
        model_to_finetune=loaded,
    )
    assert trained.pipeline

    message = Message.build(text="Rasa is great!")
    trained.featurize_message(message)
    new_sparse_feature_sizes = message.get_sparse_feature_sizes(attribute=TEXT)

    final_diet_layers = (
        trained.pipeline[-1]
        .model._tf_layers["sequence_layer.text"]
        ._tf_layers["feature_combining"]
    )
    final_diet_sequence_layer = final_diet_layers._tf_layers[
        "sparse_dense.sequence"
    ]._tf_layers["sparse_to_dense"]
    final_diet_sentence_layer = final_diet_layers._tf_layers[
        "sparse_dense.sentence"
    ]._tf_layers["sparse_to_dense"]

    final_diet_sequence_size = final_diet_sequence_layer.get_kernel().shape[0]
    final_diet_sentence_size = final_diet_sentence_layer.get_kernel().shape[0]
    assert final_diet_sequence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SEQUENCE]
    )
    assert final_diet_sentence_size == sum(
        new_sparse_feature_sizes[FEATURE_TYPE_SENTENCE]
    )
    # check if the data signatures were correctly updated
    new_data_signature = trained.pipeline[-1].model.data_signature
    new_predict_data_signature = trained.pipeline[-1].model.predict_data_signature
    iter2_data = load_data(iter2_data_path)
    expected_sequence_lengths = len(iter2_data.training_examples)

    def test_data_signatures(
        new_signature: Dict[Text, Dict[Text, List[FeatureArray]]],
        old_signature: Dict[Text, Dict[Text, List[FeatureArray]]],
    ):
        # Wherever attribute / feature_type signature is not
        # expected to change, directly compare it to old data signature.
        # Else compute its expected signature and compare
        attributes_expected_to_change = [TEXT]
        feature_types_expected_to_change = [
            FEATURE_TYPE_SEQUENCE,
            FEATURE_TYPE_SENTENCE,
        ]

        for attribute, signatures in new_signature.items():

            for feature_type, feature_signatures in signatures.items():

                if feature_type == "sequence_lengths":
                    assert feature_signatures[0].units == expected_sequence_lengths

                elif feature_type not in feature_types_expected_to_change:
                    assert feature_signatures == old_signature.get(attribute).get(
                        feature_type
                    )
                else:
                    for index, feature_signature in enumerate(feature_signatures):
                        if (
                            feature_signature.is_sparse
                            and attribute in attributes_expected_to_change
                        ):
                            assert feature_signature.units == sum(
                                new_sparse_feature_sizes.get(feature_type)
                            )
                        else:
                            # dense signature or attributes that are not
                            # expected to change can be compared directly
                            assert (
                                feature_signature.units
                                == old_signature.get(attribute)
                                .get(feature_type)[index]
                                .units
                            )

    test_data_signatures(new_data_signature, old_data_signature)
    test_data_signatures(new_predict_data_signature, old_predict_data_signature)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "iter1_path, iter2_path, should_raise_exception",
    [
        (
            "data/test_incremental_training/",
            "data/test_incremental_training/iter1",
            True,
        ),
        (
            "data/test_incremental_training/iter1",
            "data/test_incremental_training/",
            False,
        ),
    ],
)
async def test_sparse_feature_sizes_decreased_incremental_training(
    iter1_path: Text,
    iter2_path: Text,
    should_raise_exception: bool,
    component_builder: ComponentBuilder,
    tmpdir: Path,
):
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "LexicalSyntacticFeaturizer"},
        {"name": "RegexFeaturizer"},
        {"name": "CountVectorsFeaturizer"},
        {
            "name": "CountVectorsFeaturizer",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        {"name": "DIETClassifier", EPOCHS: 1},
    ]
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (_, trained, persisted_path) = await rasa.nlu.train.train(
        _config, path=str(tmpdir), data=iter1_path, component_builder=component_builder,
    )
    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder, new_config=_config,)
    assert loaded.pipeline
    assert loaded.parse("Rasa is great!") == trained.parse("Rasa is great!")
    if should_raise_exception:
        with pytest.raises(Exception) as exec_info:
            (_, trained, _) = await rasa.nlu.train.train(
                _config,
                path=str(tmpdir),
                data=iter2_path,
                component_builder=component_builder,
                model_to_finetune=loaded,
            )
        assert "Sparse feature sizes have decreased" in str(exec_info.value)
    else:
        (_, trained, _) = await rasa.nlu.train.train(
            _config,
            path=str(tmpdir),
            data=iter2_path,
            component_builder=component_builder,
            model_to_finetune=loaded,
        )
        assert trained.pipeline
