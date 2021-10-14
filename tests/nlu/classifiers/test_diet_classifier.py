from dataclasses import dataclass
import itertools
from pathlib import Path
import copy
from typing import Optional, List, Set, Text, Dict, Any, Tuple, Callable

import numpy as np
import pytest

import rasa.utils.common
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.constants import (
    BILOU_ENTITIES,
    TOKENS_NAMES,
)
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ENTITY_ATTRIBUTE_TYPE,
)
from rasa.utils.tensorflow.constants import (
    FEATURIZERS,
    LOSS_TYPE,
    MASK,
    RANDOM_SEED,
    RANKING_LENGTH,
    EPOCHS,
    MASKED_LM,
    RENORMALIZE_CONFIDENCES,
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
    HIDDEN_LAYERS_SIZES,
)
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.nlu.training_data.loading import load_data
from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.model_data_utils import FeatureArray
from rasa.utils.tensorflow.models import LABEL_KEY, LABEL_SUB_KEY
from tests.nlu.dummy_data.dummy_features import (
    ConcatenatedFeaturizations,
    DummyFeatures,
    FeaturizerDescription,
)
from tests.nlu.dummy_data.dummy_nlu_data import (
    TextIntentAndEntitiesDummy,
    IntentAndEntitiesEncodings,
)


@pytest.fixture()
def default_diet_resource() -> Resource:
    return Resource("DIET")


@pytest.fixture()
def create_diet(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    default_diet_resource: Resource,
) -> Callable[..., DIETClassifier]:
    def inner(
        config: Dict[Text, Any], load: bool = False, finetune: bool = False
    ) -> DIETClassifier:
        if load:
            constructor = DIETClassifier.load
        else:
            constructor = DIETClassifier.create

        default_execution_context.is_finetuning = finetune
        return constructor(
            config=rasa.utils.common.override_defaults(
                DIETClassifier.get_default_config(), config
            ),
            model_storage=default_model_storage,
            execution_context=default_execution_context,
            resource=default_diet_resource,
        )

    return inner


@pytest.fixture()
def create_train_load_and_process_diet(
    nlu_data_path: Text,
    create_diet: Callable[..., DIETClassifier],
    train_load_and_process_diet: Callable[..., Message],
) -> Callable[..., Message]:
    def inner(
        diet_config: Dict[Text, Any],
        pipeline: Optional[List[Dict[Text, Any]]] = None,
        training_data: str = nlu_data_path,
        message_text: Text = "Rasa is great!",
        expect_intent: bool = True,
    ) -> Message:
        diet = create_diet(diet_config)
        return train_load_and_process_diet(
            diet=diet,
            pipeline=pipeline,
            training_data=training_data,
            message_text=message_text,
            expect_intent=expect_intent,
        )

    return inner


@pytest.fixture()
def train_load_and_process_diet(
    nlu_data_path: Text,
    train_and_preprocess: Callable[..., Tuple[TrainingData, List[GraphComponent]]],
    process_message: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
    default_model_storage: ModelStorage,
) -> Callable[..., Message]:
    def inner(
        diet: DIETClassifier,
        pipeline: Optional[List[Dict[Text, Any]]] = None,
        training_data: str = nlu_data_path,
        message_text: Text = "Rasa is great!",
        expect_intent: bool = True,
    ) -> Message:

        if not pipeline:
            pipeline = [
                {"component": WhitespaceTokenizer},
                {"component": CountVectorsFeaturizer},
            ]

        training_data, loaded_pipeline = train_and_preprocess(pipeline, training_data)

        diet.train(training_data=training_data)

        message = Message(data={TEXT: message_text})
        message = process_message(loaded_pipeline, message)

        message2 = copy.deepcopy(message)

        classified_message = diet.process([message])[0]

        if expect_intent:
            assert classified_message.data["intent"]["name"]

        loaded_diet = create_diet(diet.component_config, load=True)

        classified_message2 = loaded_diet.process([message2])[0]

        assert classified_message2.fingerprint() == classified_message.fingerprint()
        return classified_message

    return inner


def test_compute_default_label_features():
    label_features = [
        Message(data={TEXT: "test a"}),
        Message(data={TEXT: "test b"}),
        Message(data={TEXT: "test c"}),
        Message(data={TEXT: "test d"}),
    ]

    output = DIETClassifier._compute_default_label_features(label_features)

    output = output[0]

    for i, o in enumerate(output):
        assert isinstance(o, np.ndarray)
        assert o[0][i] == 1
        assert o.shape == (1, len(label_features))


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
                Message(
                    data={TEXT: "test b"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                    ],
                ),
            ],
            True,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(1), FEATURE_TYPE_SEQUENCE, INTENT, "test"),
                        Features(np.zeros(1), FEATURE_TYPE_SENTENCE, INTENT, "test"),
                    ],
                )
            ],
            False,
        ),
        (
            [
                Message(
                    data={TEXT: "test a"},
                    features=[
                        Features(np.zeros(2), FEATURE_TYPE_SEQUENCE, INTENT, "test")
                    ],
                )
            ],
            False,
        ),
    ],
)
def test_check_labels_features_exist(
    messages: List[Message], expected: bool, create_diet: Callable[..., DIETClassifier]
):
    attribute = TEXT
    classifier = create_diet({})
    assert classifier._check_labels_features_exist(messages, attribute) == expected


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
                    }
                ),
                Message(
                    data={
                        TEXT: "test b",
                        INTENT: "intent b",
                        ENTITIES: [
                            {"start": 0, "end": 4, "value": "test", "entity": "test"}
                        ],
                    }
                ),
            ],
            True,
        ),
        (
            [
                Message(data={TEXT: "test a", INTENT: "intent a"}),
                Message(data={TEXT: "test b", INTENT: "intent b"}),
            ],
            False,
        ),
    ],
)
def test_model_data_signature_with_entities(
    messages: List[Message],
    entity_expected: bool,
    create_diet: Callable[..., DIETClassifier],
    whitespace_tokenizer: WhitespaceTokenizer,
):
    classifier = create_diet({"BILOU_flag": False})
    training_data = TrainingData(messages)

    # create tokens for entity parsing inside DIET
    whitespace_tokenizer.process_training_data(training_data)

    model_data = classifier.preprocess_train_data(training_data)
    entity_exists = "entities" in model_data.get_signature().keys()
    assert entity_exists == entity_expected


@pytest.mark.skip_on_windows
@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_different_settings_non_windows(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    pipeline = [
        {
            "component": WhitespaceTokenizer,
            "intent_tokenization_flag": True,
            "intent_split_symbol": "+",
        },
        {"component": CountVectorsFeaturizer},
    ]
    config = {MASKED_LM: True, EPOCHS: 1}
    create_train_load_and_process_diet(config, pipeline)
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_different_settings(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    config = {LOSS_TYPE: "margin", EPOCHS: 1}
    create_train_load_and_process_diet(config)
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(240, func_only=True)
async def test_train_persist_load_with_nested_dict_config(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    config = {HIDDEN_LAYERS_SIZES: {"text": [256, 512]}, ENTITY_RECOGNITION: False}
    create_train_load_and_process_diet(config)
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(210, func_only=True)
async def test_train_persist_load_with_only_entity_recognition(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    config = {ENTITY_RECOGNITION: True, INTENT_CLASSIFICATION: False, EPOCHS: 1}
    create_train_load_and_process_diet(
        config,
        training_data="data/examples/rasa/demo-rasa-multi-intent.yml",
        expect_intent=False,
    )
    create_diet(config, load=True, finetune=True)


@pytest.mark.timeout(120, func_only=True)
async def test_train_persist_load_with_only_intent_classification(
    create_train_load_and_process_diet: Callable[..., Message],
    create_diet: Callable[..., DIETClassifier],
):
    create_train_load_and_process_diet(
        {ENTITY_RECOGNITION: False, INTENT_CLASSIFICATION: True, EPOCHS: 1}
    )
    create_diet({MASKED_LM: True, EPOCHS: 1}, load=True, finetune=True)


@pytest.mark.parametrize(
    "classifier_params, data_path, output_length, output_should_sum_to_1",
    [
        (
            {},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            False,
        ),  # (num_intents > default ranking_length)
        (
            {RENORMALIZE_CONFIDENCES: True},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH,
            True,
        ),  # (num_intents > default ranking_length) + renormalize
        (
            {RANKING_LENGTH: 0},
            "data/test/many_intents.yml",
            16,
            True,
        ),  # (ranking_length := num_intents)
        (
            {RANKING_LENGTH: 0, RENORMALIZE_CONFIDENCES: True},
            "data/test/many_intents.yml",
            16,
            True,
        ),  # (ranking_length := num_intents) + (unnecessary) renormalize
        (
            {RANKING_LENGTH: LABEL_RANKING_LENGTH + 1},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH + 1,
            False,
        ),  # (num_intents > specified ranking_length)
        (
            {RANKING_LENGTH: LABEL_RANKING_LENGTH + 1, RENORMALIZE_CONFIDENCES: True},
            "data/test/many_intents.yml",
            LABEL_RANKING_LENGTH + 1,
            True,
        ),  # (num_intents > specified ranking_length) + renormalize
        (
            {},
            "data/test_moodbot/data/nlu.yml",
            7,
            True,
        ),  # (num_intents < default ranking_length)
    ],
)
async def test_softmax_normalization(
    classifier_params,
    data_path: Text,
    output_length,
    output_should_sum_to_1,
    create_train_load_and_process_diet: Callable[..., Message],
):
    classifier_params[RANDOM_SEED] = 42
    classifier_params[EPOCHS] = 1
    classifier_params[EVAL_NUM_EPOCHS] = 1

    parsed_message = create_train_load_and_process_diet(
        classifier_params, training_data=data_path
    )
    parse_data = parsed_message.data
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


async def test_margin_loss_is_not_normalized(
    create_train_load_and_process_diet: Callable[..., Message]
):

    parsed_message = create_train_load_and_process_diet(
        {LOSS_TYPE: "margin", RANDOM_SEED: 42, EPOCHS: 1, EVAL_NUM_EPOCHS: 1},
        training_data="data/test/many_intents.yml",
    )
    parse_data = parsed_message.data
    intent_ranking = parse_data.get("intent_ranking")

    # check that the output was correctly truncated
    assert len(intent_ranking) == LABEL_RANKING_LENGTH

    # check that output was not normalized
    assert [item["confidence"] for item in intent_ranking] != pytest.approx(1)

    # make sure top ranking is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.timeout(120, func_only=True)
async def test_set_random_seed(
    create_train_load_and_process_diet: Callable[..., Message]
):
    """test if train result is the same for two runs of tf embedding"""

    parsed_message1 = create_train_load_and_process_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 1, EPOCHS: 1}
    )

    parsed_message2 = create_train_load_and_process_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 1, EPOCHS: 1}
    )

    # Different random seed
    parsed_message3 = create_train_load_and_process_diet(
        {ENTITY_RECOGNITION: False, RANDOM_SEED: 2, EPOCHS: 1}
    )

    assert (
        parsed_message1.data["intent"]["confidence"]
        == parsed_message2.data["intent"]["confidence"]
    )
    assert (
        parsed_message2.data["intent"]["confidence"]
        != parsed_message3.data["intent"]["confidence"]
    )


@pytest.mark.parametrize("log_level", ["epoch", "batch"])
async def test_train_tensorboard_logging(
    log_level: Text,
    tmpdir: Path,
    create_train_load_and_process_diet: Callable[..., Message],
):
    tensorboard_log_dir = Path(tmpdir / "tensorboard")

    assert not tensorboard_log_dir.exists()

    pipeline = [
        {"component": WhitespaceTokenizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 3,
            "max_ngram": 17,
            "max_features": 10,
            "min_df": 5,
        },
    ]

    create_train_load_and_process_diet(
        {
            EPOCHS: 1,
            TENSORBOARD_LOG_LEVEL: log_level,
            TENSORBOARD_LOG_DIR: str(tensorboard_log_dir),
            MODEL_CONFIDENCE: "softmax",
            CONSTRAIN_SIMILARITIES: True,
            EVAL_NUM_EXAMPLES: 15,
            EVAL_NUM_EPOCHS: 1,
        },
        pipeline,
    )

    assert tensorboard_log_dir.exists()

    all_files = list(tensorboard_log_dir.rglob("*.*"))
    assert len(all_files) == 2


async def test_train_model_checkpointing(
    default_model_storage: ModelStorage,
    default_diet_resource: Resource,
    create_train_load_and_process_diet: Callable[..., Message],
):
    create_train_load_and_process_diet(
        {EPOCHS: 2, EVAL_NUM_EPOCHS: 1, EVAL_NUM_EXAMPLES: 10, CHECKPOINT_MODEL: True}
    )

    with default_model_storage.read_from(default_diet_resource) as model_dir:
        checkpoint_dir = model_dir / "checkpoints"

        assert checkpoint_dir.is_dir()

        """
        Tricky to validate the *exact* number of files that should be there, however
        there must be at least the following:
            - metadata.json
            - checkpoint
            - component_1_CountVectorsFeaturizer (as per the pipeline above)
            - component_2_DIETClassifier files (more than 1 file)
        """
        all_files = list(model_dir.rglob("*.*"))
        assert len(all_files) > 4


async def test_train_model_not_checkpointing(
    default_model_storage: ModelStorage,
    default_diet_resource: Resource,
    create_train_load_and_process_diet: Callable[..., Message],
):
    create_train_load_and_process_diet({EPOCHS: 2, CHECKPOINT_MODEL: False})

    with default_model_storage.read_from(default_diet_resource) as model_dir:
        checkpoint_dir = model_dir / "checkpoints"

        assert not checkpoint_dir.is_dir()


async def test_train_fails_with_zero_eval_num_epochs(
    create_diet: Callable[..., DIETClassifier]
):
    with pytest.raises(InvalidConfigException):
        with pytest.warns(UserWarning) as warning:
            create_diet(
                {
                    EPOCHS: 1,
                    CHECKPOINT_MODEL: True,
                    EVAL_NUM_EPOCHS: 0,
                    EVAL_NUM_EXAMPLES: 10,
                }
            )

    warn_text = (
        f"You have opted to save the best model, but the value of '{EVAL_NUM_EPOCHS}' "
        f"is not -1 or greater than 0. Training will fail."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1


async def test_doesnt_checkpoint_with_zero_eval_num_examples(
    create_diet: Callable[..., DIETClassifier],
    default_model_storage: ModelStorage,
    default_diet_resource: Resource,
    train_load_and_process_diet: Callable[..., Message],
):
    with pytest.warns(UserWarning) as warning:
        classifier = create_diet(
            {
                EPOCHS: 2,
                CHECKPOINT_MODEL: True,
                EVAL_NUM_EXAMPLES: 0,
                EVAL_NUM_EPOCHS: 1,
            }
        )

    warn_text = (
        f"You have opted to save the best model, but the value of "
        f"'{EVAL_NUM_EXAMPLES}' is not greater than 0. No checkpoint model "
        f"will be saved."
    )
    assert len([w for w in warning if warn_text in str(w.message)]) == 1

    train_load_and_process_diet(classifier)

    with default_model_storage.read_from(default_diet_resource) as model_dir:
        checkpoint_dir = model_dir / "checkpoints"

        assert not checkpoint_dir.is_dir()


@pytest.mark.parametrize(
    "classifier_params",
    [
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: False},
        {RANDOM_SEED: 1, EPOCHS: 1, BILOU_FLAG: True},
    ],
)
@pytest.mark.timeout(300, func_only=True)
async def test_train_persist_load_with_composite_entities(
    classifier_params: Dict[Text, Any],
    create_train_load_and_process_diet: Callable[..., Message],
):
    create_train_load_and_process_diet(
        classifier_params,
        training_data="data/test/demo-rasa-composite-entities.yml",
        message_text="I am looking for an italian restaurant",
    )


@pytest.mark.parametrize("should_add_diagnostic_data", [True, False])
async def test_process_gives_diagnostic_data(
    create_train_load_and_process_diet: Callable[..., Message],
    default_execution_context: ExecutionContext,
    should_add_diagnostic_data: bool,
):
    default_execution_context.should_add_diagnostic_data = should_add_diagnostic_data
    default_execution_context.node_name = "DIETClassifier_node_name"
    processed_message = create_train_load_and_process_diet({EPOCHS: 1})

    if should_add_diagnostic_data:
        # Tests if processing a message returns attention weights as numpy array.
        diagnostic_data = processed_message.get(DIAGNOSTIC_DATA)

        # DIETClassifier should add attention weights
        name = "DIETClassifier_node_name"
        assert isinstance(diagnostic_data, dict)
        assert name in diagnostic_data
        assert "attention_weights" in diagnostic_data[name]
        assert isinstance(diagnostic_data[name].get("attention_weights"), np.ndarray)
        assert "text_transformed" in diagnostic_data[name]
        assert isinstance(diagnostic_data[name].get("text_transformed"), np.ndarray)
    else:
        assert DIAGNOSTIC_DATA not in processed_message.data


@pytest.mark.parametrize(
    "initial_sparse_feature_sizes, final_sparse_feature_sizes, label_attribute",
    [
        (
            {
                TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]},
                INTENT: {FEATURE_TYPE_SEQUENCE: [5], FEATURE_TYPE_SENTENCE: []},
            },
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            INTENT,
        ),
        (
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            {TEXT: {FEATURE_TYPE_SEQUENCE: [10], FEATURE_TYPE_SENTENCE: [20]}},
            INTENT,
        ),
    ],
)
def test_removing_label_sparse_feature_sizes(
    initial_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    final_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
    label_attribute: Text,
):
    """Tests if label attribute is removed from sparse feature sizes collection."""
    feature_sizes = DIETClassifier._remove_label_sparse_feature_sizes(
        sparse_feature_sizes=initial_sparse_feature_sizes,
        label_attribute=label_attribute,
    )
    assert feature_sizes == final_sparse_feature_sizes


@pytest.mark.timeout(120)
async def test_adjusting_layers_incremental_training(
    create_diet: Callable[..., DIETClassifier],
    train_load_and_process_diet: Callable[..., Message],
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
        {"component": WhitespaceTokenizer},
        {"component": LexicalSyntacticFeaturizer},
        {"component": RegexFeaturizer},
        {"component": CountVectorsFeaturizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
    ]
    classifier = create_diet({EPOCHS: 1})
    processed_message = train_load_and_process_diet(
        classifier, pipeline=pipeline, training_data=iter1_data_path
    )

    old_data_signature = classifier.model.data_signature
    old_predict_data_signature = classifier.model.predict_data_signature
    old_sparse_feature_sizes = processed_message.get_sparse_feature_sizes(
        attribute=TEXT
    )
    initial_diet_layers = classifier.model._tf_layers["sequence_layer.text"]._tf_layers[
        "feature_combining"
    ]
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

    finetune_classifier = create_diet({EPOCHS: 1}, load=True, finetune=True)
    assert finetune_classifier.finetune_mode
    processed_message_finetuned = train_load_and_process_diet(
        finetune_classifier, pipeline=pipeline, training_data=iter2_data_path
    )

    new_sparse_feature_sizes = processed_message_finetuned.get_sparse_feature_sizes(
        attribute=TEXT
    )

    final_diet_layers = finetune_classifier.model._tf_layers[
        "sequence_layer.text"
    ]._tf_layers["feature_combining"]
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
    new_data_signature = finetune_classifier.model.data_signature
    new_predict_data_signature = finetune_classifier.model.predict_data_signature
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
    create_diet: Callable[..., DIETClassifier],
    train_load_and_process_diet: Callable[..., Message],
):
    pipeline = [
        {"component": WhitespaceTokenizer},
        {"component": LexicalSyntacticFeaturizer},
        {"component": RegexFeaturizer},
        {"component": CountVectorsFeaturizer},
        {
            "component": CountVectorsFeaturizer,
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
    ]

    classifier = create_diet({EPOCHS: 1})
    assert not classifier.finetune_mode
    train_load_and_process_diet(classifier, pipeline=pipeline, training_data=iter1_path)

    finetune_classifier = create_diet({EPOCHS: 1}, load=True, finetune=True)
    assert finetune_classifier.finetune_mode

    if should_raise_exception:
        with pytest.raises(Exception) as exec_info:
            train_load_and_process_diet(
                finetune_classifier, pipeline=pipeline, training_data=iter2_path
            )
        assert "Sparse feature sizes have decreased" in str(exec_info.value)
    else:
        train_load_and_process_diet(
            finetune_classifier, pipeline=pipeline, training_data=iter2_path
        )


class TextIntentAndEntitiesTestRuns:
    """Describes all test cases using the `TextIntentAndEntitiesDummy`."""

    CONFIGURATIONS = [
        {
            "component_config": {
                INTENT_CLASSIFICATION: intent_classification,
                ENTITY_RECOGNITION: entity_recognition,
                FEATURIZERS: used_featurizers,
                BILOU_FLAG: bilou_tagging,
                EPOCHS: 1,
                CONSTRAIN_SIMILARITIES: True,
            },
            "input_contains_features_for_intent": featurize_intent,
            "attributes_without_sentence_features": skip_sentence_features,
        }
        for skip_sentence_features in [set(), {TEXT}, {INTENT}, {TEXT, INTENT}]
        for used_featurizers in [None, ["2", "1"]]  # switch order of used featurizers
        for (
            featurize_intent,
            intent_classification,
            entity_recognition,
            bilou_tagging,
        ) in itertools.product([True, False], repeat=4)
        # skip if there's no goal
        if (intent_classification or entity_recognition)
        # skip bilou tagging without entities enabled
        and not (bilou_tagging and not entity_recognition)
        # skip intent featurization tests if we don't do intent featurization
        and not (not intent_classification and featurize_intent)
        # skip when we want to skip sentence features for intents but there are none
        and not (INTENT in skip_sentence_features and featurize_intent)
        # skip some more (no interesting combinations)
        and not (len(skip_sentence_features) > 0 and used_featurizers is None)
    ]

    @dataclass
    class TestRunData:
        # generator used to create dummy data and features
        meta: TextIntentAndEntitiesDummy
        # dummy data used for this test run
        messages: List[Message]
        # features created for each message as used by the intent classifier
        # (i.e. features of `used_featurizers` are concatenated per feature type and
        # sparseness property)
        concatenated_features: Dict[Text, List[ConcatenatedFeaturizations]]
        trained_model: DIETClassifier
        # Each test run prepares data for training, `label_data` and prediction.
        # We collect the results for each of these tasks separately, i.e. the
        # used messages (inputs), resulting rasa model data (output) and the
        # expected results (e.g. which indices from the input are to be used).
        sub_results: Dict[
            Text, Tuple[List[Message], RasaModelData, IntentAndEntitiesEncodings]
        ]

    @staticmethod
    def run(
        component_config: Dict[Text, Any],
        input_contains_features_for_intent: bool,
        attributes_without_sentence_features: Set[Text],
    ) -> TestRunData:
        """Creates a model and prepares `TextIntentAndEntitiesData` dummy data."""
        # Create dummy data
        dummy_data = TextIntentAndEntitiesDummy(
            featurize_intents=input_contains_features_for_intent,
            no_sentence_features=attributes_without_sentence_features,
        )
        original_messages = dummy_data.create_messages()
        dummy_data.featurize_messages(original_messages)
        concatenated_features = dummy_data.create_and_concatenate_features(
            messages=original_messages, used_featurizers=component_config[FEATURIZERS],
        )
        expected_results = dummy_data.intent_classifier_usage()

        # Create the model
        model = create_diet(config=component_config)

        # Preprocess training data
        messages_for_training = copy.deepcopy(original_messages)
        model_data_for_training = model.preprocess_train_data(
            training_data=TrainingData(messages_for_training)
        )

        # Imitate creation of model data during inference time
        messages_for_prediction = copy.deepcopy(original_messages)
        model_data_for_prediction = model._create_model_data(
            messages_for_prediction, training=False,
        )

        return TextIntentAndEntitiesTestRuns.TestRunData(
            meta=dummy_data,
            messages=original_messages,
            concatenated_features=concatenated_features,
            trained_model=model,
            sub_results={
                "training": (
                    messages_for_training,
                    model_data_for_training,
                    expected_results["training"],
                ),
                "label_data": (
                    original_messages,
                    model._label_data,
                    expected_results["label_data"],
                ),
                "prediction": (
                    messages_for_prediction,
                    model_data_for_prediction,
                    expected_results["prediction"],
                ),
            },
        )


@pytest.mark.parametrize("config", TextIntentAndEntitiesTestRuns.CONFIGURATIONS)
def test_create_data_label_mapping(config: Dict[Text, Any]):
    # 1. Note that this mapping is needed by predict, so we test this separately.
    # 2. Observe that the current version *always* computes this mapping, even if
    #    intent classification is switched off.
    test_run = TextIntentAndEntitiesTestRuns.run(**config)
    _, _, expected_result = test_run.sub_results["label_data"]
    # Note: we strip spaces here
    expected = {
        intent_index: test_run.messages[message_index].get(INTENT).strip()
        for intent_index, message_index in enumerate(expected_result.indices)
    }
    assert test_run.trained_model.index_label_id_mapping == expected


@pytest.mark.parametrize(
    "config, key",
    [
        (config, key)
        for config in TextIntentAndEntitiesTestRuns.CONFIGURATIONS
        for key in ["training", "prediction"]
    ],
)
def test_create_data_subkey_text(config: Dict[Text, Any], key: Text):

    # test run
    test_run = TextIntentAndEntitiesTestRuns.run(**config)
    used_messages, model_data, expected_results = test_run.sub_results[key]

    # subkeys for TEXT
    expected_text_sub_keys = {MASK, SEQUENCE, SEQUENCE_LENGTH}
    if TEXT not in config["attributes_without_sentence_features"]:
        expected_text_sub_keys.add(SENTENCE)
    assert set(model_data.get(TEXT).keys()) == expected_text_sub_keys
    text_features = model_data.get(TEXT)

    # subkey: mask (this is a "turn" mask)
    mask_features_array = text_features.get(MASK)
    assert len(mask_features_array) == 1
    mask = np.array(mask_features_array[0])
    assert mask.shape == (len(expected_results.indices), 1, 1)
    assert np.all(
        mask.flatten()
        == [
            1.0 if used_messages[idx].get(TEXT, None) else 0.0
            for idx in expected_results.indices
        ]
    )

    # subkey: sequence-length
    length_features_array = text_features.get(SEQUENCE_LENGTH)
    assert len(length_features_array) == 1
    lengths = np.array(length_features_array[0])
    assert lengths.shape == (len(expected_results.indices),)
    expected_lengths = [
        len(used_messages[idx].get(TOKENS_NAMES[TEXT], []))
        for idx in expected_results.indices
    ]
    assert np.all(lengths == expected_lengths)

    # subkey: sequence and possibly sentence (see above)
    DummyFeatures.compare_with_feature_arrays(
        expected=[
            test_run.concatenated_features[TEXT][idx]
            for idx in expected_results.indices
        ],
        actual=text_features,
    )

    # sparse feature sizes
    used_featurizer_names = config["component_config"][FEATURIZERS]
    all_featurizers: List[FeaturizerDescription] = test_run.meta.featurizers
    sparse_used_featurizers = [
        d
        for d in all_featurizers
        if d.is_sparse
        and ((used_featurizer_names is None) or (d.name in used_featurizer_names))
    ]
    expected_sizes = [d.dimension for d in sparse_used_featurizers]
    actual_sizes_by_type = model_data.sparse_feature_sizes[TEXT]
    feature_types = [FEATURE_TYPE_SEQUENCE]
    if SENTENCE in expected_text_sub_keys:
        feature_types.append(FEATURE_TYPE_SENTENCE)
    for feature_type in feature_types:
        assert np.all(actual_sizes_by_type[feature_type] == expected_sizes)


@pytest.mark.parametrize(
    "config, key",
    [
        (config, key)
        for config in TextIntentAndEntitiesTestRuns.CONFIGURATIONS
        for key in ["training", "label_data"]
        if config["component_config"].get(INTENT_CLASSIFICATION)
    ],
)
def test_create_data_subkey_label(config: Dict[Text, Any], key: Text):

    # test run
    test_run = TextIntentAndEntitiesTestRuns.run(**config)
    input_contains_features_for_intent = config["input_contains_features_for_intent"]
    used_messages, model_data, expected_results = test_run.sub_results[key]
    has_mask_features = key != "label_data"

    # subkeys for label
    expected_label_subkeys = {LABEL_SUB_KEY}
    if has_mask_features:
        expected_label_subkeys.add(MASK)
    if input_contains_features_for_intent:
        expected_label_subkeys.update({SEQUENCE, SEQUENCE_LENGTH})
        if INTENT not in config["attributes_without_sentence_features"]:
            expected_label_subkeys.add(SENTENCE)
    else:
        expected_label_subkeys.update({SENTENCE})
    label_features = model_data.get(LABEL_KEY)
    assert set(label_features.keys()) == expected_label_subkeys

    # subkey: mask (this is a "turn" mask, hence all masks are just "[1]")
    if has_mask_features:
        assert len(label_features.get(MASK)) == 1
        mask = np.array(label_features.get(MASK)[0])
        assert np.all(mask == [1] * len(expected_results.indices))

    # subkey: label_sub_key / id
    id_features_array = label_features.get(LABEL_SUB_KEY)
    assert len(id_features_array) == 1
    id_features = np.array(id_features_array[0])
    assert np.all(id_features == [[id] for id in expected_results.intents_ids])

    # - subkey: sentence
    if not input_contains_features_for_intent:
        # - subkey: sequence and sequence length
        assert SEQUENCE not in label_features
        assert SEQUENCE_LENGTH not in label_features
        # - subkey: sentence
        sentence_feature_array = label_features.get(SENTENCE)
        assert len(sentence_feature_array) == 1
        generated_label_features = np.array(sentence_feature_array[0])
        assert np.all(generated_label_features == expected_results.intents_mhot)
    else:
        # - subkey: sequence-length
        length_feature_array = label_features.get(SEQUENCE_LENGTH)
        assert len(length_feature_array) == 1
        lengths = np.array(length_feature_array[0])
        expected_lengths = [
            len(used_messages[idx].get(TOKENS_NAMES[INTENT], []))
            for idx in expected_results.indices
        ]
        assert np.all(lengths == expected_lengths)
        # - subkeys: sequence and possibly sentence (see above)
        DummyFeatures.compare_with_feature_arrays(
            expected=[
                test_run.concatenated_features[INTENT][idx]
                for idx in expected_results.indices
            ],
            actual=label_features,
        )


@pytest.mark.parametrize(
    "config, key",
    [
        (config, key)
        for config in TextIntentAndEntitiesTestRuns.CONFIGURATIONS
        for key in ["training", "prediction"]
        if config["component_config"].get(ENTITY_RECOGNITION)
    ],
)
def test_create_data_subkey_entities(config: Dict[Text, Any], key: Text):

    # test run
    test_run = TextIntentAndEntitiesTestRuns.run(**config)
    component_config = config["component_config"]
    used_messages, model_data, expected_results = test_run.sub_results["training"]

    # sub-keys for ENTITIES
    expected_entity_sub_keys = {ENTITY_ATTRIBUTE_TYPE, MASK}
    entity_features = model_data.get(ENTITIES)
    assert entity_features
    assert set(entity_features.keys()) == expected_entity_sub_keys
    for key in entity_features:
        assert len(entity_features.get(key)) == 1
        assert isinstance(entity_features.get(key)[0], FeatureArray)

    # subkey: mask  (this is a "turn" mask, hence all masks are just "[1]")
    mask = np.array(entity_features.get(MASK)[0])
    expected_mask = [1] * len(used_messages)
    assert np.all(mask == expected_mask)

    # subkey: entities
    entity_sub_features = entity_features[ENTITY_ATTRIBUTE_TYPE][0]
    assert len(entity_sub_features) == len(expected_results.indices)
    for idx, orig_idx in enumerate(expected_results.indices):
        if component_config.get(BILOU_FLAG):
            assert np.all(
                np.array(entity_sub_features[idx]).flatten()
                == expected_results.entities_bilou_ids[idx]
            )
            assert (
                used_messages[orig_idx].get(BILOU_ENTITIES)
                == expected_results.entities_bilou_tags[idx]
            )
        else:
            assert np.all(
                np.array(entity_sub_features[idx]).flatten()
                == expected_results.entities_ids[idx]
            )
            assert not used_messages[orig_idx].get(BILOU_ENTITIES)
