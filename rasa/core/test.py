import logging
import os
import warnings
import typing
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.core.channels import UserMessage
from rasa.core.training.story_writer.yaml_story_writer import YAMLStoryWriter
import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.nlu.constants import (
    ENTITIES,
    EXTRACTOR,
    ENTITY_ATTRIBUTE_VALUE,
    ENTITY_ATTRIBUTE_TEXT,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    ENTITY_ATTRIBUTE_TYPE,
    INTENT,
)
from rasa.constants import RESULTS_FILE, PERCENTAGE_KEY
from rasa.core.utils import pad_lists_to_size
from rasa.core.events import ActionExecuted, UserUttered
from rasa.core.trackers import DialogueStateTracker
from rasa.nlu.training_data.formats.readerwriter import TrainingDataWriter
from rasa.utils.io import DEFAULT_ENCODING

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent
    from rasa.core.processor import MessageProcessor


CONFUSION_MATRIX_STORIES_FILE = "story_confusion_matrix.png"
REPORT_STORIES_FILE = "story_report.json"
FAILED_STORIES_FILE = "failed_test_stories.yml"
SUCCESSFUL_STORIES_FILE = "successful_test_stories.yml"


logger = logging.getLogger(__name__)

StoryEvaluation = namedtuple(
    "StoryEvaluation",
    [
        "evaluation_store",
        "failed_stories",
        "successful_stories",
        "action_list",
        "in_training_data_fraction",
    ],
)


class EvaluationStore:
    """Class storing action, intent and entity predictions and targets."""

    def __init__(
        self,
        action_predictions: Optional[List[Text]] = None,
        action_targets: Optional[List[Text]] = None,
        intent_predictions: Optional[List[Text]] = None,
        intent_targets: Optional[List[Text]] = None,
        entity_predictions: Optional[List[Dict[Text, Any]]] = None,
        entity_targets: Optional[List[Dict[Text, Any]]] = None,
    ) -> None:
        self.action_predictions = action_predictions or []
        self.action_targets = action_targets or []
        self.intent_predictions = intent_predictions or []
        self.intent_targets = intent_targets or []
        self.entity_predictions = entity_predictions or []
        self.entity_targets = entity_targets or []

    def add_to_store(
        self,
        action_predictions: Optional[List[Text]] = None,
        action_targets: Optional[List[Text]] = None,
        intent_predictions: Optional[List[Text]] = None,
        intent_targets: Optional[List[Text]] = None,
        entity_predictions: Optional[List[Dict[Text, Any]]] = None,
        entity_targets: Optional[List[Dict[Text, Any]]] = None,
    ) -> None:
        """Add items or lists of items to the store"""

        self.action_predictions.extend(action_predictions or [])
        self.action_targets.extend(action_targets or [])
        self.intent_targets.extend(intent_targets or [])
        self.intent_predictions.extend(intent_predictions or [])
        self.entity_predictions.extend(entity_predictions or [])
        self.entity_targets.extend(entity_targets or [])

    def merge_store(self, other: "EvaluationStore") -> None:
        """Add the contents of other to self"""
        self.add_to_store(
            action_predictions=other.action_predictions,
            action_targets=other.action_targets,
            intent_predictions=other.intent_predictions,
            intent_targets=other.intent_targets,
            entity_predictions=other.entity_predictions,
            entity_targets=other.entity_targets,
        )

    def has_prediction_target_mismatch(self) -> bool:
        return (
            self.intent_predictions != self.intent_targets
            or self.entity_predictions != self.entity_targets
            or self.action_predictions != self.action_targets
        )

    def serialise(self) -> Tuple[List[Text], List[Text]]:
        """Turn targets and predictions to lists of equal size for sklearn."""

        targets = (
            self.action_targets
            + self.intent_targets
            + [
                TrainingDataWriter.generate_entity(gold.get("text"), gold)
                for gold in self.entity_targets
            ]
        )
        predictions = (
            self.action_predictions
            + self.intent_predictions
            + [
                TrainingDataWriter.generate_entity(predicted.get("text"), predicted)
                for predicted in self.entity_predictions
            ]
        )

        # sklearn does not cope with lists of unequal size, nor None values
        return pad_lists_to_size(targets, predictions, padding_value="None")


class WronglyPredictedAction(ActionExecuted):
    """The model predicted the wrong action.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_action"

    def __init__(
        self,
        action_name_target: Text,
        action_name_prediction: Text,
        policy: Optional[Text] = None,
        confidence: Optional[float] = None,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.action_name_prediction = action_name_prediction
        super().__init__(action_name_target, policy, confidence, timestamp, metadata)

    def inline_comment(self) -> Text:
        """A comment attached to this event. Used during dumping."""
        return f"predicted: {self.action_name_prediction}"

    def as_story_string(self) -> Text:
        return f"{self.action_name}   <!-- {self.inline_comment()} -->"


class EndToEndUserUtterance(UserUttered):
    """End-to-end user utterance.

    Mostly used to print the full end-to-end user message in the
    `failed_test_stories.yml` output file."""

    def as_story_string(self, e2e: bool = True) -> Text:
        return super().as_story_string(e2e=True)


class WronglyClassifiedUserUtterance(UserUttered):
    """The NLU model predicted the wrong user utterance.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_utterance"

    def __init__(self, event: UserUttered, eval_store: EvaluationStore) -> None:

        if not eval_store.intent_predictions:
            self.predicted_intent = None
        else:
            self.predicted_intent = eval_store.intent_predictions[0]
        self.predicted_entities = eval_store.entity_predictions

        intent = {"name": eval_store.intent_targets[0]}

        super().__init__(
            event.text,
            intent,
            eval_store.entity_targets,
            event.parse_data,
            event.timestamp,
            event.input_channel,
        )

    def inline_comment(self) -> Text:
        """A comment attached to this event. Used during dumping."""
        from rasa.core.events import md_format_message

        predicted_message = md_format_message(
            self.text, self.predicted_intent, self.predicted_entities
        )
        return f"predicted: {self.predicted_intent}: {predicted_message}"

    def as_story_string(self, e2e: bool = True) -> Text:
        from rasa.core.events import md_format_message

        correct_message = md_format_message(
            self.text, self.intent.get("name"), self.entities
        )
        return (
            f"{self.intent.get('name')}: {correct_message}   "
            f"<!-- {self.inline_comment()} -->"
        )


async def _generate_trackers(
    resource_name: Text,
    agent: "Agent",
    max_stories: Optional[int] = None,
    use_e2e: bool = False,
) -> List[Any]:
    from rasa.core.training.generator import TrainingDataGenerator

    from rasa.core import training

    story_graph = await training.extract_story_graph(
        resource_name, agent.domain, use_e2e
    )
    g = TrainingDataGenerator(
        story_graph,
        agent.domain,
        use_story_concatenation=False,
        augmentation_factor=0,
        tracker_limit=max_stories,
    )
    return g.generate_story_trackers()


def _clean_entity_results(
    text: Text, entity_results: List[Dict[Text, Any]]
) -> List[Dict[Text, Any]]:
    """Extract only the token variables from an entity dict."""
    cleaned_entities = []

    for r in tuple(entity_results):
        cleaned_entity = {ENTITY_ATTRIBUTE_TEXT: text}
        for k in (
            ENTITY_ATTRIBUTE_START,
            ENTITY_ATTRIBUTE_END,
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_VALUE,
        ):
            if k in set(r):
                if k == ENTITY_ATTRIBUTE_VALUE and EXTRACTOR in set(r):
                    # convert values to strings for evaluation as
                    # target values are all of type string
                    r[k] = str(r[k])
                cleaned_entity[k] = r[k]
        cleaned_entities.append(cleaned_entity)

    return cleaned_entities


def _collect_user_uttered_predictions(
    event: UserUttered,
    predicted: Dict[Text, Any],
    partial_tracker: DialogueStateTracker,
    fail_on_prediction_errors: bool,
) -> EvaluationStore:
    user_uttered_eval_store = EvaluationStore()

    intent_gold = event.intent.get("name")
    predicted_intent = predicted.get(INTENT, {}).get("name")

    user_uttered_eval_store.add_to_store(
        intent_predictions=[predicted_intent], intent_targets=[intent_gold]
    )

    entity_gold = event.entities
    predicted_entities = predicted.get(ENTITIES)

    if entity_gold or predicted_entities:
        user_uttered_eval_store.add_to_store(
            entity_targets=_clean_entity_results(event.text, entity_gold),
            entity_predictions=_clean_entity_results(event.text, predicted_entities),
        )

    if user_uttered_eval_store.has_prediction_target_mismatch():
        partial_tracker.update(
            WronglyClassifiedUserUtterance(event, user_uttered_eval_store)
        )
        if fail_on_prediction_errors:
            raise ValueError(
                "NLU model predicted a wrong intent. Failed Story:"
                " \n\n{}".format(
                    YAMLStoryWriter().dumps(partial_tracker.as_story().story_steps)
                )
            )
    else:
        end_to_end_user_utterance = EndToEndUserUtterance(
            event.text, event.intent, event.entities
        )
        partial_tracker.update(end_to_end_user_utterance)

    return user_uttered_eval_store


def _emulate_form_rejection(partial_tracker: DialogueStateTracker) -> None:
    from rasa.core.events import ActionExecutionRejected

    rejected_action_name: Text = partial_tracker.active_loop["name"]
    partial_tracker.update(ActionExecutionRejected(rejected_action_name))


def _collect_action_executed_predictions(
    processor: "MessageProcessor",
    partial_tracker: DialogueStateTracker,
    event: ActionExecuted,
    fail_on_prediction_errors: bool,
    circuit_breaker_tripped: bool,
) -> Tuple[EvaluationStore, Optional[Text], Optional[float]]:
    from rasa.core.policies.form_policy import FormPolicy

    action_executed_eval_store = EvaluationStore()

    gold = event.action_name

    if circuit_breaker_tripped:
        predicted = "circuit breaker tripped"
        policy = None
        confidence = None
    else:
        action, policy, confidence = processor.predict_next_action(partial_tracker)
        predicted = action.name()

        if (
            policy
            and predicted != gold
            and _form_might_have_been_rejected(
                processor.domain, partial_tracker, predicted
            )
        ):
            # Wrong action was predicted,
            # but it might be Ok if form action is rejected.
            _emulate_form_rejection(partial_tracker)
            # try again
            action, policy, confidence = processor.predict_next_action(partial_tracker)

            # Even if the prediction is also wrong, we don't have to undo the emulation
            # of the action rejection as we know that the user explicitly specified
            # that something else than the form was supposed to run.
            predicted = action.name()

    action_executed_eval_store.add_to_store(
        action_predictions=[predicted], action_targets=[gold]
    )

    if action_executed_eval_store.has_prediction_target_mismatch():
        partial_tracker.update(
            WronglyPredictedAction(
                gold, predicted, event.policy, event.confidence, event.timestamp
            )
        )
        if fail_on_prediction_errors:
            error_msg = (
                "Model predicted a wrong action. Failed Story: "
                "\n\n{}".format(
                    YAMLStoryWriter().dumps(partial_tracker.as_story().story_steps)
                )
            )
            if FormPolicy.__name__ in policy:
                error_msg += (
                    "FormAction is not run during "
                    "evaluation therefore it is impossible to know "
                    "if validation failed or this story is wrong. "
                    "If the story is correct, add it to the "
                    "training stories and retrain."
                )
            raise ValueError(error_msg)
    else:
        partial_tracker.update(event)

    return action_executed_eval_store, policy, confidence


def _form_might_have_been_rejected(
    domain: Domain, tracker: DialogueStateTracker, predicted_action_name: Text
) -> bool:
    return (
        tracker.active_loop.get("name") == predicted_action_name
        and predicted_action_name in domain.form_names
    )


async def _predict_tracker_actions(
    tracker: DialogueStateTracker,
    agent: "Agent",
    fail_on_prediction_errors: bool = False,
    use_e2e: bool = False,
) -> Tuple[EvaluationStore, DialogueStateTracker, List[Dict[Text, Any]]]:

    processor = agent.create_processor()
    tracker_eval_store = EvaluationStore()

    events = list(tracker.events)

    partial_tracker = DialogueStateTracker.from_events(
        tracker.sender_id,
        events[:1],
        agent.domain.slots,
        sender_source=tracker.sender_source,
    )

    tracker_actions = []
    should_predict_another_action = True
    num_predicted_actions = 0

    for event in events[1:]:
        if isinstance(event, ActionExecuted):
            circuit_breaker_tripped = processor.is_action_limit_reached(
                num_predicted_actions, should_predict_another_action
            )
            (
                action_executed_result,
                policy,
                confidence,
            ) = _collect_action_executed_predictions(
                processor,
                partial_tracker,
                event,
                fail_on_prediction_errors,
                circuit_breaker_tripped,
            )
            tracker_eval_store.merge_store(action_executed_result)
            tracker_actions.append(
                {
                    "action": action_executed_result.action_targets[0],
                    "predicted": action_executed_result.action_predictions[0],
                    "policy": policy,
                    "confidence": confidence,
                }
            )
            should_predict_another_action = processor.should_predict_another_action(
                action_executed_result.action_predictions[0]
            )
            num_predicted_actions += 1

        elif use_e2e and isinstance(event, UserUttered):
            predicted = await processor.parse_message(UserMessage(event.text))
            user_uttered_result = _collect_user_uttered_predictions(
                event, predicted, partial_tracker, fail_on_prediction_errors
            )

            tracker_eval_store.merge_store(user_uttered_result)
        else:
            partial_tracker.update(event)
        if isinstance(event, UserUttered):
            num_predicted_actions = 0

    return tracker_eval_store, partial_tracker, tracker_actions


def _in_training_data_fraction(action_list: List[Dict[Text, Any]]) -> float:
    """Given a list of action items, returns the fraction of actions

    that were predicted using one of the Memoization policies."""
    from rasa.core.policies.ensemble import SimplePolicyEnsemble

    in_training_data = [
        a["action"]
        for a in action_list
        if a["policy"] and not SimplePolicyEnsemble.is_not_memo_policy(a["policy"])
    ]

    return len(in_training_data) / len(action_list) if action_list else 0


async def _collect_story_predictions(
    completed_trackers: List["DialogueStateTracker"],
    agent: "Agent",
    fail_on_prediction_errors: bool = False,
    use_e2e: bool = False,
) -> Tuple[StoryEvaluation, int]:
    """Test the stories from a file, running them through the stored model."""
    from rasa.test import get_evaluation_metrics
    from tqdm import tqdm

    story_eval_store = EvaluationStore()
    failed = []
    success = []
    correct_dialogues = []
    number_of_stories = len(completed_trackers)

    logger.info(f"Evaluating {number_of_stories} stories\nProgress:")

    action_list = []

    for tracker in tqdm(completed_trackers):
        (
            tracker_results,
            predicted_tracker,
            tracker_actions,
        ) = await _predict_tracker_actions(
            tracker, agent, fail_on_prediction_errors, use_e2e
        )

        story_eval_store.merge_store(tracker_results)

        action_list.extend(tracker_actions)

        if tracker_results.has_prediction_target_mismatch():
            # there is at least one wrong prediction
            failed.append(predicted_tracker)
            correct_dialogues.append(0)
        else:
            correct_dialogues.append(1)
            success.append(predicted_tracker)

    logger.info("Finished collecting predictions.")
    with warnings.catch_warnings():
        from sklearn.exceptions import UndefinedMetricWarning

        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report, precision, f1, accuracy = get_evaluation_metrics(
            [1] * len(completed_trackers), correct_dialogues
        )

    in_training_data_fraction = _in_training_data_fraction(action_list)

    _log_evaluation_table(
        [1] * len(completed_trackers),
        "END-TO-END" if use_e2e else "CONVERSATION",
        report,
        precision,
        f1,
        accuracy,
        in_training_data_fraction,
        include_report=False,
    )

    return (
        StoryEvaluation(
            evaluation_store=story_eval_store,
            failed_stories=failed,
            successful_stories=success,
            action_list=action_list,
            in_training_data_fraction=in_training_data_fraction,
        ),
        number_of_stories,
    )


def _log_stories(trackers: List[DialogueStateTracker], file_path: Text) -> None:
    """Write given stories to the given file."""

    with open(file_path, "w", encoding=DEFAULT_ENCODING) as f:
        if not trackers:
            f.write("# None of the test stories failed - all good!")
        else:
            stories = [tracker.as_story(include_source=True) for tracker in trackers]
            steps = [step for story in stories for step in story.story_steps]
            f.write(YAMLStoryWriter().dumps(steps))


async def test(
    stories: Text,
    agent: "Agent",
    max_stories: Optional[int] = None,
    out_directory: Optional[Text] = None,
    fail_on_prediction_errors: bool = False,
    e2e: bool = False,
    disable_plotting: bool = False,
    successes: bool = False,
    errors: bool = True,
) -> Dict[Text, Any]:
    """Run the evaluation of the stories, optionally plot the results.

    Args:
        stories: the stories to evaluate on
        agent: the agent
        max_stories: maximum number of stories to consider
        out_directory: path to directory to results to
        fail_on_prediction_errors: boolean indicating whether to fail on prediction
            errors or not
        e2e: boolean indicating whether to use end to end evaluation or not
        disable_plotting: boolean indicating whether to disable plotting or not
        successes: boolean indicating whether to write down successful predictions or
            not
        errors: boolean indicating whether to write down incorrect predictions or not

    Returns:
        Evaluation summary.
    """
    from rasa.test import get_evaluation_metrics

    completed_trackers = await _generate_trackers(stories, agent, max_stories, e2e)

    story_evaluation, _ = await _collect_story_predictions(
        completed_trackers, agent, fail_on_prediction_errors, e2e
    )

    evaluation_store = story_evaluation.evaluation_store

    with warnings.catch_warnings():
        from sklearn.exceptions import UndefinedMetricWarning

        warnings.simplefilter("ignore", UndefinedMetricWarning)

        targets, predictions = evaluation_store.serialise()

        if out_directory:
            report, precision, f1, accuracy = get_evaluation_metrics(
                targets, predictions, output_dict=True
            )

            report_filename = os.path.join(out_directory, REPORT_STORIES_FILE)
            io_utils.dump_obj_as_json_to_file(report_filename, report)
            logger.info(f"Stories report saved to {report_filename}.")
        else:
            report, precision, f1, accuracy = get_evaluation_metrics(
                targets, predictions, output_dict=True
            )

    _log_evaluation_table(
        evaluation_store.action_targets,
        "ACTION",
        report,
        precision,
        f1,
        accuracy,
        story_evaluation.in_training_data_fraction,
        include_report=False,
    )

    if not disable_plotting and out_directory:
        _plot_story_evaluation(
            evaluation_store.action_targets,
            evaluation_store.action_predictions,
            out_directory,
        )

    if errors and out_directory:
        _log_stories(
            story_evaluation.failed_stories,
            os.path.join(out_directory, FAILED_STORIES_FILE),
        )
    if successes and out_directory:
        _log_stories(
            story_evaluation.successful_stories,
            os.path.join(out_directory, SUCCESSFUL_STORIES_FILE),
        )

    return {
        "report": report,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "actions": story_evaluation.action_list,
        "in_training_data_fraction": story_evaluation.in_training_data_fraction,
        "is_end_to_end_evaluation": e2e,
    }


def _log_evaluation_table(
    golds: List[Any],
    name: Text,
    report: Dict[Text, Any],
    precision: float,
    f1: float,
    accuracy: float,
    in_training_data_fraction: float,
    include_report: bool = True,
) -> None:  # pragma: no cover
    """Log the sklearn evaluation metrics."""
    logger.info(f"Evaluation Results on {name} level:")
    logger.info(f"\tCorrect:          {int(len(golds) * accuracy)} / {len(golds)}")
    logger.info(f"\tF1-Score:         {f1:.3f}")
    logger.info(f"\tPrecision:        {precision:.3f}")
    logger.info(f"\tAccuracy:         {accuracy:.3f}")
    logger.info(f"\tIn-data fraction: {in_training_data_fraction:.3g}")

    if include_report:
        logger.info(f"\tClassification report: \n{report}")


def _plot_story_evaluation(
    targets: List[Text], predictions: List[Text], output_directory: Optional[Text]
) -> None:
    """Plot a confusion matrix of story evaluation."""
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    from rasa.utils.plotting import plot_confusion_matrix

    confusion_matrix_filename = CONFUSION_MATRIX_STORIES_FILE
    if output_directory:
        confusion_matrix_filename = os.path.join(
            output_directory, confusion_matrix_filename
        )

    cnf_matrix = confusion_matrix(targets, predictions)

    plot_confusion_matrix(
        cnf_matrix,
        classes=unique_labels(targets, predictions),
        title="Action Confusion matrix",
        output_file=confusion_matrix_filename,
    )


async def compare_models_in_dir(
    model_dir: Text, stories_file: Text, output: Text
) -> None:
    """Evaluate multiple trained models in a directory on a test set.

    Args:
        model_dir: path to directory that contains the models to evaluate
        stories_file: path to the story file
        output: output directory to store results to
    """
    number_correct = defaultdict(list)

    for run in io_utils.list_subdirectories(model_dir):
        number_correct_in_run = defaultdict(list)

        for model in sorted(io_utils.list_files(run)):
            if not model.endswith("tar.gz"):
                continue

            # The model files are named like <config-name>PERCENTAGE_KEY<number>.tar.gz
            # Remove the percentage key and number from the name to get the config name
            config_name = os.path.basename(model).split(PERCENTAGE_KEY)[0]
            number_of_correct_stories = await _evaluate_core_model(model, stories_file)
            number_correct_in_run[config_name].append(number_of_correct_stories)

        for k, v in number_correct_in_run.items():
            number_correct[k].append(v)

    io_utils.dump_obj_as_json_to_file(
        os.path.join(output, RESULTS_FILE), number_correct
    )


async def compare_models(models: List[Text], stories_file: Text, output: Text) -> None:
    """Evaluate provided trained models on a test set.

    Args:
        models: list of trained model paths
        stories_file: path to the story file
        output: output directory to store results to
    """
    number_correct = defaultdict(list)

    for model in models:
        number_of_correct_stories = await _evaluate_core_model(model, stories_file)
        number_correct[os.path.basename(model)].append(number_of_correct_stories)

    io_utils.dump_obj_as_json_to_file(
        os.path.join(output, RESULTS_FILE), number_correct
    )


async def _evaluate_core_model(model: Text, stories_file: Text) -> int:
    from rasa.core.agent import Agent

    logger.info(f"Evaluating model '{model}'")

    agent = Agent.load(model)
    completed_trackers = await _generate_trackers(stories_file, agent)
    story_eval_store, number_of_stories = await _collect_story_predictions(
        completed_trackers, agent
    )
    failed_stories = story_eval_store.failed_stories
    return number_of_stories - len(failed_stories)


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.core.test` directly is no longer supported. Please use "
        "`rasa test` to test a combined Core and NLU model or `rasa test core` "
        "to test a Core model."
    )
