import asyncio
import sys
from pathlib import Path
import textwrap
from typing import List, Text

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

import rasa.shared.utils.io
import rasa.utils.io
from rasa.core.agent import Agent
from rasa.shared.core.events import UserUttered
from rasa.core.test import (
    EvaluationStore,
    WronglyClassifiedUserUtterance,
    WronglyPredictedAction,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.training_data.story_writer.yaml_story_writer import (
    YAMLStoryWriter,
)
import rasa.model
import rasa.cli.utils
from rasa.nlu.test import NO_ENTITY
import rasa.core
from rasa.shared.nlu.constants import (
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_TEXT,
)
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION


def monkeypatch_get_latest_model(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    latest_model = tmp_path / "my_test_model.tar.gz"
    monkeypatch.setattr(rasa.model, "get_latest_model", lambda: str(latest_model))


def test_get_sanitized_model_directory_when_not_passing_model(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.model_testing import _get_sanitized_model_directory

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    # Create a fake model on disk so that `is_file` returns `True`
    latest_model = Path(rasa.model.get_latest_model())
    latest_model.touch()

    # Input: default model file
    # => Should return containing directory
    new_modeldir = _get_sanitized_model_directory(str(latest_model))
    captured = capsys.readouterr()
    assert not captured.out
    assert new_modeldir == str(latest_model.parent)


def test_get_sanitized_model_directory_when_passing_model_file_explicitly(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.model_testing import _get_sanitized_model_directory

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    other_model = tmp_path / "my_test_model1.tar.gz"
    assert str(other_model) != rasa.model.get_latest_model()
    other_model.touch()

    # Input: some file
    # => Should return containing directory and print a warning
    new_modeldir = _get_sanitized_model_directory(str(other_model))
    captured = capsys.readouterr()
    assert captured.out
    assert new_modeldir == str(other_model.parent)


def test_get_sanitized_model_directory_when_passing_other_input(
    capsys: CaptureFixture, tmp_path: Path, monkeypatch: MonkeyPatch
):
    from rasa.model_testing import _get_sanitized_model_directory

    monkeypatch_get_latest_model(tmp_path, monkeypatch)

    # Input: anything that is not an existing file
    # => Should return input
    modeldir = "random_dir"
    assert not Path(modeldir).is_file()
    new_modeldir = _get_sanitized_model_directory(modeldir)
    captured = capsys.readouterr()
    assert not captured.out
    assert new_modeldir == modeldir


@pytest.mark.parametrize(
    "targets,predictions,expected_precision,expected_fscore,expected_accuracy",
    [
        (
            ["no_entity", "location", "no_entity", "location", "no_entity"],
            ["no_entity", "location", "no_entity", "no_entity", "person"],
            1.0,
            0.6666666666666666,
            3 / 5,
        ),
        (
            ["no_entity", "no_entity", "no_entity", "no_entity", "person"],
            ["no_entity", "no_entity", "no_entity", "no_entity", "no_entity"],
            0.0,
            0.0,
            4 / 5,
        ),
    ],
)
def test_get_evaluation_metrics(
    targets: List[Text],
    predictions: List[Text],
    expected_precision: float,
    expected_fscore: float,
    expected_accuracy: float,
):
    from rasa.model_testing import get_evaluation_metrics

    report, precision, f1, accuracy = get_evaluation_metrics(
        targets, predictions, True, exclude_label=NO_ENTITY
    )

    assert f1 == expected_fscore
    assert precision == expected_precision
    assert accuracy == expected_accuracy
    assert NO_ENTITY not in report


@pytest.mark.parametrize(
    "report_in,accuracy,report_out",
    [
        (
            {
                "location": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "micro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "macro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "weighted avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
            },
            0.8,
            {
                "location": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "micro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "macro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "weighted avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "accuracy": 0.8,
            },
        ),
        (
            {
                "location": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "macro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "weighted avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "accuracy": 0.8,
            },
            0.8,
            {
                "location": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "micro avg": {
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1-score": 0.8,
                    "support": 2,
                },
                "macro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "weighted avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "accuracy": 0.8,
            },
        ),
    ],
)
def test_make_classification_report_complete(
    report_in: dict, accuracy: float, report_out: dict
):
    from rasa.model_testing import make_classification_report_complete

    report_out_actual = make_classification_report_complete(report_in, accuracy)
    assert report_out == report_out_actual


@pytest.mark.parametrize(
    "report_in",
    [
        (
            {
                "location": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "micro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "macro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "weighted avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "accuracy": 0.8,
            },
        ),
        (
            {
                "location": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "macro avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
                "weighted avg": {
                    "precision": 1.0,
                    "recall": 0.5,
                    "f1-score": 0.6666666666666666,
                    "support": 2,
                },
            },
        ),
    ],
)
def test_make_classification_report_complete_raises_clf_report_exception(
    report_in: dict,
):
    from rasa.model_testing import (
        ClassificationReportException,
        make_classification_report_complete,
    )

    with pytest.raises(ClassificationReportException):
        make_classification_report_complete(report_in, accuracy=0.8)


@pytest.mark.parametrize(
    "targets,exclude_label,expected",
    [
        (
            ["no_entity", "location", "location", "location", "person"],
            NO_ENTITY,
            ["location", "person"],
        ),
        (
            ["no_entity", "location", "location", "location", "person"],
            None,
            ["no_entity", "location", "person"],
        ),
        (["no_entity"], NO_ENTITY, []),
        (["location", "location", "location"], NO_ENTITY, ["location"]),
        ([], None, []),
    ],
)
def test_get_label_set(targets: List[Text], exclude_label: Text, expected: List[Text]):
    from rasa.model_testing import get_unique_labels

    actual = get_unique_labels(targets, exclude_label)
    assert set(expected) == set(actual)


async def test_e2e_warning_if_no_nlu_model(
    monkeypatch: MonkeyPatch, trained_core_model: Text, capsys: CaptureFixture
):
    from rasa.model_testing import test_core

    # Patching is bit more complicated as we have a module `train` and function
    # with the same name 😬
    monkeypatch.setattr(
        sys.modules["rasa.core.test"], "test", asyncio.coroutine(lambda *_, **__: True)
    )

    await test_core(trained_core_model, use_conversation_test_files=True)

    assert "No NLU model found. Using default" in capsys.readouterr().out


def test_write_classification_errors():
    evaluation = EvaluationStore(
        action_predictions=["utter_goodbye"],
        action_targets=["utter_greet"],
        intent_predictions=["goodbye"],
        intent_targets=["greet"],
        entity_predictions=None,
        entity_targets=None,
    )
    events = [
        WronglyClassifiedUserUtterance(
            UserUttered("Hello", {"name": "goodbye"}), evaluation
        ),
        WronglyPredictedAction("utter_greet", "", "utter_goodbye"),
    ]
    tracker = DialogueStateTracker.from_events("default", events)
    dump = YAMLStoryWriter().dumps(tracker.as_story().story_steps)
    assert (
        dump.strip()
        == textwrap.dedent(
            f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        stories:
        - story: default
          steps:
          - intent: greet  # predicted: goodbye: Hello
          - action: utter_greet  # predicted: utter_goodbye

    """
        ).strip()
    )


def test_log_failed_stories(tmp_path: Path):
    path = str(tmp_path / "stories.yml")
    rasa.core.test._log_stories([], path, "Some text")

    dump = rasa.shared.utils.io.read_file(path)

    assert dump.startswith("#")
    assert len(dump.split("\n")) == 1


@pytest.mark.parametrize(
    "entity_predictions,entity_targets",
    [
        (
            [{"text": "hi, how are you", "start": 4, "end": 7, "entity": "aa"}],
            [
                {"text": "hi, how are you", "start": 0, "end": 2, "entity": "bb"},
                {"text": "hi, how are you", "start": 4, "end": 7, "entity": "aa"},
            ],
        ),
        (
            [
                {"text": "hi, how are you", "start": 0, "end": 2, "entity": "bb"},
                {"text": "hi, how are you", "start": 4, "end": 7, "entity": "aa"},
            ],
            [
                {"text": "hi, how are you", "start": 0, "end": 2, "entity": "bb"},
                {"text": "hi, how are you", "start": 4, "end": 7, "entity": "aa"},
            ],
        ),
        (
            [
                {"text": "hi, how are you", "start": 0, "end": 2, "entity": "bb"},
                {"text": "hi, how are you", "start": 4, "end": 7, "entity": "aa"},
            ],
            [{"text": "hi, how are you", "start": 4, "end": 7, "entity": "aa"}],
        ),
        (
            [
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 0,
                    "end": 5,
                    "entity": "person",
                },
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 22,
                    "end": 28,
                    "entity": "city",
                },
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 47,
                    "end": 53,
                    "entity": "city",
                },
            ],
            [
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 22,
                    "end": 28,
                    "entity": "city",
                }
            ],
        ),
        (
            [
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 0,
                    "end": 5,
                    "entity": "person",
                },
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 47,
                    "end": 53,
                    "entity": "city",
                },
            ],
            [
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 22,
                    "end": 28,
                    "entity": "city",
                },
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 47,
                    "end": 53,
                    "entity": "city",
                },
            ],
        ),
        (
            [
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 47,
                    "end": 53,
                    "entity": "city",
                }
            ],
            [
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 0,
                    "end": 5,
                    "entity": "person",
                },
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 22,
                    "end": 28,
                    "entity": "city",
                },
                {
                    "text": "Tanja is currently in Munich, but she lives in Berlin",
                    "start": 47,
                    "end": 53,
                    "entity": "city",
                },
            ],
        ),
    ],
)
def test_evaluation_store_serialise(
    entity_predictions: List[dict], entity_targets: List[dict]
):
    from rasa.shared.nlu.training_data.formats.readerwriter import TrainingDataWriter

    store = EvaluationStore(
        entity_predictions=entity_predictions, entity_targets=entity_targets
    )

    targets, predictions = store.serialise()

    assert len(targets) == len(predictions)

    i_pred = 0
    i_target = 0
    for i, prediction in enumerate(predictions):
        target = targets[i]
        if prediction != "None" and target != "None":
            predicted = entity_predictions[i_pred]
            assert prediction == TrainingDataWriter.generate_entity(
                predicted.get("text"), predicted
            )
            assert predicted.get("start") == entity_targets[i_target].get("start")
            assert predicted.get("end") == entity_targets[i_target].get("end")

        if prediction != "None":
            i_pred += 1
        if target != "None":
            i_target += 1


def test_test_does_not_use_rules(tmp_path: Path, default_agent: Agent):
    from rasa.core.test import _create_data_generator

    test_file = tmp_path / "test.yml"
    test_name = "my test story"
    tests = f"""
stories:
- story: {test_name}
  steps:
  - intent: greet
  - action: utter_greet

rules:
- rule: rule which is ignored
  steps:
  - intent: greet
  - action: utter_greet
    """

    test_file.write_text(tests)

    generator = _create_data_generator(str(test_file), default_agent)
    test_trackers = generator.generate_story_trackers()
    assert len(test_trackers) == 1
    assert test_trackers[0].sender_id == test_name


def test_duplicated_entity_predictions_tolerated():
    """Same entity extracted multiple times shouldn't be flagged as prediction error.

    This can happen when multiple entity extractors extract the same entity but a test
    story only lists the entity once. For completeness, the other case (entity listed
    twice in test story and extracted once) is also tested here because it should work
    the same way.
    """
    entity = {
        ENTITY_ATTRIBUTE_TEXT: "Algeria",
        ENTITY_ATTRIBUTE_START: 0,
        ENTITY_ATTRIBUTE_END: 7,
        ENTITY_ATTRIBUTE_VALUE: "Algeria",
        ENTITY_ATTRIBUTE_TYPE: "country",
    }
    evaluation_with_duplicated_prediction = EvaluationStore(
        entity_predictions=[entity, entity], entity_targets=[entity]
    )
    assert not evaluation_with_duplicated_prediction.check_prediction_target_mismatch()

    evaluation_with_duplicated_target = EvaluationStore(
        entity_predictions=[entity], entity_targets=[entity, entity]
    )
    assert not evaluation_with_duplicated_target.check_prediction_target_mismatch()


def test_differently_ordered_entity_predictions_tolerated():
    """The order in which entities were extracted shouldn't matter.

    Let's have an utterance like this: "[Researcher](job_name) from [Germany](country)."
    and imagine we use different entity extractors for the two entities. Then, the order
    in which entities are extracted from the utterance depends on the order in which the
    extractors are listed in the NLU pipeline. However, the expected order is given by
    where the entities are found in the utterance, i.e. "Researcher" comes before
    "Germany". Hence, it's reasonable for the expected and extracted order to not match
    and it shouldn't be flagged as a prediction error.

    """
    entity1 = {
        ENTITY_ATTRIBUTE_TEXT: "Algeria and Albania",
        ENTITY_ATTRIBUTE_START: 0,
        ENTITY_ATTRIBUTE_END: 7,
        ENTITY_ATTRIBUTE_VALUE: "Algeria",
        ENTITY_ATTRIBUTE_TYPE: "country",
    }
    entity2 = {
        ENTITY_ATTRIBUTE_TEXT: "Algeria and Albania",
        ENTITY_ATTRIBUTE_START: 12,
        ENTITY_ATTRIBUTE_END: 19,
        ENTITY_ATTRIBUTE_VALUE: "Albania",
        ENTITY_ATTRIBUTE_TYPE: "country",
    }
    evaluation = EvaluationStore(
        entity_predictions=[entity1, entity2], entity_targets=[entity2, entity1]
    )
    assert not evaluation.check_prediction_target_mismatch()
