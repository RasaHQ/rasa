from pathlib import Path

import pytest

from rasa.nlu import train
from rasa.nlu.components import ComponentBuilder
from rasa.shared.nlu.training_data import util
from rasa.nlu.config import RasaNLUModelConfig
import rasa.shared.nlu.training_data.loading
from rasa.nlu.train import Trainer, Interpreter
from rasa.utils.tensorflow.constants import (
    EPOCHS,
    MASKED_LM,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    EVAL_NUM_EPOCHS,
    EVAL_NUM_EXAMPLES,
    CHECKPOINT_MODEL,
)
from rasa.nlu.selectors.response_selector import ResponseSelector
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData


@pytest.mark.parametrize(
    "pipeline",
    [
        [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {"name": "ResponseSelector", EPOCHS: 1},
        ],
        [
            {"name": "WhitespaceTokenizer"},
            {"name": "CountVectorsFeaturizer"},
            {
                "name": "ResponseSelector",
                EPOCHS: 1,
                MASKED_LM: True,
                TRANSFORMER_SIZE: 256,
                NUM_TRANSFORMER_LAYERS: 1,
            },
        ],
    ],
)
def test_train_selector(pipeline, component_builder, tmpdir):
    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.md"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.md"
    )
    training_data = training_data.merge(training_data_responses)

    nlu_config = RasaNLUModelConfig({"language": "en", "pipeline": pipeline})

    trainer = Trainer(nlu_config)
    trainer.train(training_data)

    persisted_path = trainer.persist(tmpdir)

    assert trainer.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)
    parsed = loaded.parse("hello")

    assert loaded.pipeline
    assert parsed is not None
    assert (parsed.get("response_selector").get("all_retrieval_intents")) == [
        "chitchat"
    ]
    assert (
        parsed.get("response_selector")
        .get("default")
        .get("response")
        .get("intent_response_key")
    ) is not None
    assert (
        parsed.get("response_selector")
        .get("default")
        .get("response")
        .get("template_name")
    ) is not None
    assert (
        parsed.get("response_selector")
        .get("default")
        .get("response")
        .get("response_templates")
    ) is not None

    ranking = parsed.get("response_selector").get("default").get("ranking")
    assert ranking is not None

    for rank in ranking:
        assert rank.get("confidence") is not None
        assert rank.get("intent_response_key") is not None


def test_preprocess_selector_multiple_retrieval_intents():

    # use some available data
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.md"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.md"
    )
    training_data_extra_intent = TrainingData(
        [
            Message.build(
                text="Is it possible to detect the version?", intent="faq/q1"
            ),
            Message.build(text="How can I get a new virtual env", intent="faq/q2"),
        ]
    )
    training_data = training_data.merge(training_data_responses).merge(
        training_data_extra_intent
    )

    response_selector = ResponseSelector()

    response_selector.preprocess_train_data(training_data)

    assert sorted(response_selector.all_retrieval_intents) == ["chitchat", "faq"]


@pytest.mark.parametrize(
    "use_text_as_label, label_values",
    [
        [False, ["chitchat/ask_name", "chitchat/ask_weather"]],
        [True, ["I am Mr. Bot", "It's sunny where I live"]],
    ],
)
def test_ground_truth_for_training(use_text_as_label, label_values):

    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.md"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.md"
    )
    training_data = training_data.merge(training_data_responses)

    response_selector = ResponseSelector(
        component_config={"use_text_as_label": use_text_as_label}
    )
    response_selector.preprocess_train_data(training_data)

    assert response_selector.responses == training_data.responses
    assert (
        sorted(list(response_selector.index_label_id_mapping.values())) == label_values
    )


@pytest.mark.parametrize(
    "predicted_label, train_on_text, resolved_intent_response_key",
    [
        ["chitchat/ask_name", False, "chitchat/ask_name"],
        ["It's sunny where I live", True, "chitchat/ask_weather"],
    ],
)
def test_resolve_intent_response_key_from_label(
    predicted_label, train_on_text, resolved_intent_response_key
):

    # use data that include some responses
    training_data = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa.md"
    )
    training_data_responses = rasa.shared.nlu.training_data.loading.load_data(
        "data/examples/rasa/demo-rasa-responses.md"
    )
    training_data = training_data.merge(training_data_responses)

    response_selector = ResponseSelector(
        component_config={"use_text_as_label": train_on_text}
    )
    response_selector.preprocess_train_data(training_data)

    label_intent_response_key = response_selector._resolve_intent_response_key(
        {"id": hash(predicted_label), "name": predicted_label}
    )
    assert resolved_intent_response_key == label_intent_response_key
    assert (
        response_selector.responses[
            util.intent_response_key_to_template_key(label_intent_response_key)
        ]
        == training_data.responses[
            util.intent_response_key_to_template_key(resolved_intent_response_key)
        ]
    )


async def test_train_model_checkpointing(
    component_builder: ComponentBuilder, tmpdir: Path
):
    from pathlib import Path

    model_name = "rs-checkpointed-model"
    best_model_file = Path(str(tmpdir), model_name)
    assert not best_model_file.exists()

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "CountVectorsFeaturizer"},
                {
                    "name": "ResponseSelector",
                    EPOCHS: 5,
                    EVAL_NUM_EXAMPLES: 10,
                    EVAL_NUM_EPOCHS: 1,
                    CHECKPOINT_MODEL: True,
                },
            ],
            "language": "en",
        }
    )

    await train(
        _config,
        path=str(tmpdir),
        data="data/test_selectors",
        component_builder=component_builder,
        fixed_model_name=model_name,
    )

    assert best_model_file.exists()

    """
    Tricky to validate the *exact* number of files that should be there, however there must be at least the following:
        - metadata.json
        - checkpoint
        - component_1_CountVectorsFeaturizer (as per the pipeline above)
        - component_2_ResponseSelector files (more than 1 file)
    """
    all_files = list(best_model_file.rglob("*.*"))
    assert len(all_files) > 4
