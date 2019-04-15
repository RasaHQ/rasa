import argparse
import asyncio
import json
import logging
import os
import typing
import warnings
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.core.events import ActionExecuted, UserUttered

if typing.TYPE_CHECKING:
    from rasa.core.agent import Agent
    from rasa.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)

StoryEvalution = namedtuple(
    "StoryEvaluation",
    "evaluation_store failed_stories action_list in_training_data_fraction",
)


def create_argument_parser():
    """Create argument parser for the evaluate script."""
    import rasa.core.cli.arguments

    import rasa.core.cli.train
    from rasa.core import cli

    parser = argparse.ArgumentParser(description="evaluates a dialogue model")
    parent_parser = argparse.ArgumentParser(add_help=False)
    cli.test.add_evaluation_arguments(parent_parser)
    cli.arguments.add_model_and_story_group(parent_parser, allow_pretrained_model=False)
    rasa.core.cli.arguments.add_logging_option_arguments(parent_parser)
    subparsers = parser.add_subparsers(help="mode", dest="mode")
    subparsers.add_parser(
        "default",
        help="default mode: evaluate a dialogue model",
        parents=[parent_parser],
    )
    subparsers.add_parser(
        "compare",
        help="compare mode: evaluate multiple"
        " dialogue models to compare "
        "policies",
        parents=[parent_parser],
    )

    return parser


class EvaluationStore(object):
    """Class storing action, intent and entity predictions and targets."""

    def __init__(
        self,
        action_predictions: Optional[List[str]] = None,
        action_targets: Optional[List[str]] = None,
        intent_predictions: Optional[List[str]] = None,
        intent_targets: Optional[List[str]] = None,
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
        action_predictions: Optional[List[str]] = None,
        action_targets: Optional[List[str]] = None,
        intent_predictions: Optional[List[str]] = None,
        intent_targets: Optional[List[str]] = None,
        entity_predictions: Optional[List[Dict[Text, Any]]] = None,
        entity_targets: Optional[List[Dict[Text, Any]]] = None,
    ) -> None:
        """Add items or lists of items to the store"""
        for k, v in locals().items():
            if k != "self" and v:
                attr = getattr(self, k)
                if isinstance(v, list):
                    attr.extend(v)
                else:
                    attr.append(v)

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

    def has_prediction_target_mismatch(self):
        return (
            self.intent_predictions != self.intent_targets
            or self.entity_predictions != self.entity_targets
            or self.action_predictions != self.action_targets
        )

    def serialise_targets(
        self, include_actions=True, include_intents=True, include_entities=False
    ):
        targets = []
        if include_actions:
            targets += self.action_targets
        if include_intents:
            targets += self.intent_targets
        if include_entities:
            targets += self.entity_targets

        return [json.dumps(t) if isinstance(t, dict) else t for t in targets]

    def serialise_predictions(
        self, include_actions=True, include_intents=True, include_entities=False
    ):
        predictions = []

        if include_actions:
            predictions += self.action_predictions
        if include_intents:
            predictions += self.intent_predictions
        if include_entities:
            predictions += self.entity_predictions

        return [json.dumps(t) if isinstance(t, dict) else t for t in predictions]


class WronglyPredictedAction(ActionExecuted):
    """The model predicted the wrong action.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_action"

    def __init__(
        self, correct_action, predicted_action, policy, confidence, timestamp=None
    ):
        self.predicted_action = predicted_action
        super(WronglyPredictedAction, self).__init__(
            correct_action, policy, confidence, timestamp=timestamp
        )

    def as_story_string(self):
        return "{}   <!-- predicted: {} -->".format(
            self.action_name, self.predicted_action
        )


class EndToEndUserUtterance(UserUttered):
    """End-to-end user utterance.

    Mostly used to print the full end-to-end user message in the
    `failed_stories.md` output file."""

    def as_story_string(self, e2e=True):
        return super(EndToEndUserUtterance, self).as_story_string(e2e=True)


class WronglyClassifiedUserUtterance(UserUttered):
    """The NLU model predicted the wrong user utterance.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_utterance"

    def __init__(
        self,
        text,
        correct_intent,
        correct_entities,
        parse_data=None,
        timestamp=None,
        input_channel=None,
        predicted_intent=None,
        predicted_entities=None,
    ):
        self.predicted_intent = predicted_intent
        self.predicted_entities = predicted_entities

        intent = {"name": correct_intent}

        super(WronglyClassifiedUserUtterance, self).__init__(
            text, intent, correct_entities, parse_data, timestamp, input_channel
        )

    def as_story_string(self, e2e=True):
        from rasa.core.events import md_format_message

        correct_message = md_format_message(self.text, self.intent, self.entities)
        predicted_message = md_format_message(
            self.text, self.predicted_intent, self.predicted_entities
        )
        return ("{}: {}   <!-- predicted: {}: {} -->").format(
            self.intent.get("name"),
            correct_message,
            self.predicted_intent,
            predicted_message,
        )


async def _generate_trackers(resource_name, agent, max_stories=None, use_e2e=False):
    from rasa.core.training.generator import TrainingDataGenerator

    from rasa.core import training

    story_graph = await training.extract_story_graph(
        resource_name, agent.domain, agent.interpreter, use_e2e
    )
    g = TrainingDataGenerator(
        story_graph,
        agent.domain,
        use_story_concatenation=False,
        augmentation_factor=0,
        tracker_limit=max_stories,
    )
    return g.generate()


def _clean_entity_results(entity_results):
    return [
        {k: r[k] for k in ("start", "end", "entity", "value") if k in r}
        for r in entity_results
    ]


def _collect_user_uttered_predictions(
    event, partial_tracker, fail_on_prediction_errors
):
    from rasa.core.utils import pad_list_to_size

    user_uttered_eval_store = EvaluationStore()

    intent_gold = event.parse_data.get("true_intent")
    predicted_intent = event.parse_data.get("intent").get("name")
    if predicted_intent is None:
        predicted_intent = "None"
    user_uttered_eval_store.add_to_store(
        intent_predictions=predicted_intent, intent_targets=intent_gold
    )

    entity_gold = event.parse_data.get("true_entities")
    predicted_entities = event.parse_data.get("entities")

    if entity_gold or predicted_entities:
        if len(entity_gold) > len(predicted_entities):
            predicted_entities = pad_list_to_size(
                predicted_entities, len(entity_gold), "None"
            )
        elif len(predicted_entities) > len(entity_gold):
            entity_gold = pad_list_to_size(entity_gold, len(predicted_entities), "None")

        user_uttered_eval_store.add_to_store(
            entity_targets=_clean_entity_results(entity_gold),
            entity_predictions=_clean_entity_results(predicted_entities),
        )

    if user_uttered_eval_store.has_prediction_target_mismatch():
        partial_tracker.update(
            WronglyClassifiedUserUtterance(
                event.text,
                intent_gold,
                user_uttered_eval_store.entity_predictions,
                event.parse_data,
                event.timestamp,
                event.input_channel,
                predicted_intent,
                user_uttered_eval_store.entity_targets,
            )
        )
        if fail_on_prediction_errors:
            raise ValueError(
                "NLU model predicted a wrong intent. Failed Story:"
                " \n\n{}".format(partial_tracker.export_stories())
            )
    else:
        end_to_end_user_utterance = EndToEndUserUtterance(
            event.text, event.intent, event.entities
        )
        partial_tracker.update(end_to_end_user_utterance)

    return user_uttered_eval_store


def _emulate_form_rejection(processor, partial_tracker):
    from rasa.core.policies.form_policy import FormPolicy
    from rasa.core.events import ActionExecutionRejected

    if partial_tracker.active_form.get("name"):
        for p in processor.policy_ensemble.policies:
            if isinstance(p, FormPolicy):
                # emulate form rejection
                partial_tracker.update(
                    ActionExecutionRejected(partial_tracker.active_form["name"])
                )
                # check if unhappy path is covered by the train stories
                if not p.state_is_unhappy(partial_tracker, processor.domain):
                    # this state is not covered by the stories
                    del partial_tracker.events[-1]
                    partial_tracker.active_form["rejected"] = False


def _collect_action_executed_predictions(
    processor, partial_tracker, event, fail_on_prediction_errors
):
    from rasa.core.policies.form_policy import FormPolicy

    action_executed_eval_store = EvaluationStore()

    gold = event.action_name

    action, policy, confidence = processor.predict_next_action(partial_tracker)
    predicted = action.name()

    if policy and predicted != gold and FormPolicy.__name__ in policy:
        # FormPolicy predicted wrong action
        # but it might be Ok if form action is rejected
        _emulate_form_rejection(processor, partial_tracker)
        # try again
        action, policy, confidence = processor.predict_next_action(partial_tracker)
        predicted = action.name()

    action_executed_eval_store.add_to_store(
        action_predictions=predicted, action_targets=gold
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
                "\n\n{}".format(partial_tracker.export_stories())
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


def _predict_tracker_actions(
    tracker, agent: "Agent", fail_on_prediction_errors=False, use_e2e=False
):
    from rasa.core.trackers import DialogueStateTracker

    processor = agent.create_processor()
    tracker_eval_store = EvaluationStore()

    events = list(tracker.events)

    partial_tracker = DialogueStateTracker.from_events(
        tracker.sender_id, events[:1], agent.domain.slots
    )

    tracker_actions = []

    for event in events[1:]:
        if isinstance(event, ActionExecuted):
            action_executed_result, policy, confidence = _collect_action_executed_predictions(
                processor, partial_tracker, event, fail_on_prediction_errors
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
        elif use_e2e and isinstance(event, UserUttered):
            user_uttered_result = _collect_user_uttered_predictions(
                event, partial_tracker, fail_on_prediction_errors
            )

            tracker_eval_store.merge_store(user_uttered_result)
        else:
            partial_tracker.update(event)

    return tracker_eval_store, partial_tracker, tracker_actions


def _in_training_data_fraction(action_list):
    """Given a list of action items, returns the fraction of actions

    that were predicted using one of the Memoization policies."""
    from rasa.core.policies import SimplePolicyEnsemble

    in_training_data = [
        a["action"]
        for a in action_list
        if a["policy"] and not SimplePolicyEnsemble.is_not_memo_policy(a["policy"])
    ]

    return len(in_training_data) / len(action_list)


def collect_story_predictions(
    completed_trackers: List["DialogueStateTracker"],
    agent: "Agent",
    fail_on_prediction_errors: bool = False,
    use_e2e: bool = False,
) -> Tuple[StoryEvalution, int]:
    """Test the stories from a file, running them through the stored model."""
    from rasa.nlu.test import get_evaluation_metrics
    from tqdm import tqdm

    story_eval_store = EvaluationStore()
    failed = []
    correct_dialogues = []
    num_stories = len(completed_trackers)

    logger.info("Evaluating {} stories\nProgress:".format(num_stories))

    action_list = []

    for tracker in tqdm(completed_trackers):
        tracker_results, predicted_tracker, tracker_actions = _predict_tracker_actions(
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

    logger.info("Finished collecting predictions.")
    with warnings.catch_warnings():
        from sklearn.exceptions import UndefinedMetricWarning

        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report, precision, f1, accuracy = get_evaluation_metrics(
            [1] * len(completed_trackers), correct_dialogues
        )

    in_training_data_fraction = _in_training_data_fraction(action_list)

    log_evaluation_table(
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
        StoryEvalution(
            evaluation_store=story_eval_store,
            failed_stories=failed,
            action_list=action_list,
            in_training_data_fraction=in_training_data_fraction,
        ),
        num_stories,
    )


def log_failed_stories(failed, out_directory):
    """Take stories as a list of dicts."""
    if not out_directory:
        return
    with open(
        os.path.join(out_directory, "failed_stories.md"), "w", encoding="utf-8"
    ) as f:
        if len(failed) == 0:
            f.write("<!-- All stories passed -->")
        else:
            for failure in failed:
                f.write(failure.export_stories())
                f.write("\n\n")


async def test(
    stories: Text,
    agent: "Agent",
    max_stories: Optional[int] = None,
    out_directory: Optional[Text] = None,
    fail_on_prediction_errors: bool = False,
    use_e2e: bool = False,
):
    """Run the evaluation of the stories, optionally plot the results."""
    from rasa.nlu.test import get_evaluation_metrics

    completed_trackers = await _generate_trackers(stories, agent, max_stories, use_e2e)

    story_evaluation, _ = collect_story_predictions(
        completed_trackers, agent, fail_on_prediction_errors, use_e2e
    )

    evaluation_store = story_evaluation.evaluation_store

    with warnings.catch_warnings():
        from sklearn.exceptions import UndefinedMetricWarning

        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report, precision, f1, accuracy = get_evaluation_metrics(
            evaluation_store.serialise_targets(),
            evaluation_store.serialise_predictions(),
        )

    if out_directory:
        plot_story_evaluation(
            evaluation_store.action_targets,
            evaluation_store.action_predictions,
            report,
            precision,
            f1,
            accuracy,
            story_evaluation.in_training_data_fraction,
            out_directory,
        )

    log_failed_stories(story_evaluation.failed_stories, out_directory)

    return {
        "report": report,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy,
        "actions": story_evaluation.action_list,
        "in_training_data_fraction": story_evaluation.in_training_data_fraction,
        "is_end_to_end_evaluation": use_e2e,
    }


def log_evaluation_table(
    golds,
    name,
    report,
    precision,
    f1,
    accuracy,
    in_training_data_fraction,
    include_report=True,
):  # pragma: no cover
    """Log the sklearn evaluation metrics."""
    logger.info("Evaluation Results on {} level:".format(name))
    logger.info(
        "\tCorrect:          {} / {}".format(int(len(golds) * accuracy), len(golds))
    )
    logger.info("\tF1-Score:         {:.3f}".format(f1))
    logger.info("\tPrecision:        {:.3f}".format(precision))
    logger.info("\tAccuracy:         {:.3f}".format(accuracy))
    logger.info("\tIn-data fraction: {:.3g}".format(in_training_data_fraction))

    if include_report:
        logger.info("\tClassification report: \n{}".format(report))


def plot_story_evaluation(
    test_y,
    predictions,
    report,
    precision,
    f1,
    accuracy,
    in_training_data_fraction,
    out_directory,
):
    """Plot the results of story evaluation"""
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt
    from rasa.nlu.test import plot_confusion_matrix

    log_evaluation_table(
        test_y,
        "ACTION",
        report,
        precision,
        f1,
        accuracy,
        in_training_data_fraction,
        include_report=True,
    )

    cnf_matrix = confusion_matrix(test_y, predictions)

    plot_confusion_matrix(
        cnf_matrix,
        classes=unique_labels(test_y, predictions),
        title="Action Confusion matrix",
    )

    fig = plt.gcf()
    fig.set_size_inches(int(20), int(20))
    fig.savefig(os.path.join(out_directory, "story_confmat.pdf"), bbox_inches="tight")


async def compare(models: Text, stories_file: Text, output: Text) -> None:
    """Evaluates multiple trained models on a test set."""
    from rasa.core.agent import Agent
    import rasa.nlu.utils as nlu_utils
    from rasa.core import utils

    num_correct = defaultdict(list)

    for run in nlu_utils.list_subdirectories(models):
        num_correct_run = defaultdict(list)

        for model in sorted(nlu_utils.list_subdirectories(run)):
            logger.info("Evaluating model {}".format(model))

            agent = Agent.load(model)

            completed_trackers = await _generate_trackers(stories_file, agent)

            story_eval_store, no_of_stories = collect_story_predictions(
                completed_trackers, agent
            )

            failed_stories = story_eval_store.failed_stories
            policy_name = "".join(
                [i for i in os.path.basename(model) if not i.isdigit()]
            )
            num_correct_run[policy_name].append(no_of_stories - len(failed_stories))

        for k, v in num_correct_run.items():
            num_correct[k].append(v)

    utils.dump_obj_as_json_to_file(os.path.join(output, "results.json"), num_correct)


def plot_curve(output: Text, no_stories: List[int]) -> None:
    """Plot the results from run_comparison_evaluation.

    Args:
        output: Output directory to save resulting plots to
        no_stories: Number of stories per run
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from rasa.core import utils

    ax = plt.gca()

    # load results from file
    data = utils.read_json_file(os.path.join(output, "results.json"))
    x = no_stories

    # compute mean of all the runs for keras/embed policies
    for label in data.keys():
        if len(data[label]) == 0:
            continue
        mean = np.mean(data[label], axis=0)
        std = np.std(data[label], axis=0)
        ax.plot(x, mean, label=label, marker=".")
        ax.fill_between(
            x,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            color="#6b2def",
            alpha=0.2,
        )
    ax.legend(loc=4)
    ax.set_xlabel("Number of stories present during training")
    ax.set_ylabel("Number of correct test stories")
    plt.savefig(os.path.join(output, "model_comparison_graph.pdf"), format="pdf")
    plt.show()


def main():
    from rasa.core.agent import Agent
    from rasa.core.interpreter import NaturalLanguageInterpreter
    from rasa.core.utils import AvailableEndpoints, set_default_subparser
    import rasa.nlu.utils as nlu_utils
    import rasa.core.cli
    from rasa.core import utils

    loop = asyncio.get_event_loop()

    # Running as standalone python application
    arg_parser = create_argument_parser()
    set_default_subparser(arg_parser, "default")
    cmdline_arguments = arg_parser.parse_args()

    logging.basicConfig(level=cmdline_arguments.loglevel)
    _endpoints = AvailableEndpoints.read_endpoints(cmdline_arguments.endpoints)

    if cmdline_arguments.output:
        nlu_utils.create_dir(cmdline_arguments.output)

    if not cmdline_arguments.core:
        raise ValueError(
            "you must provide a core model directory to evaluate using -d / --core"
        )
    if cmdline_arguments.mode == "default":

        _interpreter = NaturalLanguageInterpreter.create(
            cmdline_arguments.nlu, _endpoints.nlu
        )

        _agent = Agent.load(cmdline_arguments.core, interpreter=_interpreter)

        stories = loop.run_until_complete(
            rasa.core.cli.train.stories_from_cli_args(cmdline_arguments)
        )

        loop.run_until_complete(
            test(
                stories,
                _agent,
                cmdline_arguments.max_stories,
                cmdline_arguments.output,
                cmdline_arguments.fail_on_prediction_errors,
                cmdline_arguments.e2e,
            )
        )

    elif cmdline_arguments.mode == "compare":
        compare(
            cmdline_arguments.core, cmdline_arguments.stories, cmdline_arguments.output
        )

        story_n_path = os.path.join(cmdline_arguments.core, "num_stories.json")

        number_of_stories = utils.read_json_file(story_n_path)
        plot_curve(cmdline_arguments.output, number_of_stories)

    logger.info("Finished evaluation")


if __name__ == "__main__":
    raise RuntimeError(
        "Calling `rasa.core.test` directly is "
        "no longer supported. "
        "Please use `rasa test core` instead."
    )
