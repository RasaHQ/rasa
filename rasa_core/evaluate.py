from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import io
import json
import logging
import warnings
from builtins import str
from typing import List, Tuple, Optional, Any, Text, Dict

from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from rasa_core import training
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.events import ActionExecuted, UserUttered
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.generator import TrainingDataGenerator
from rasa_core.utils import AvailableEndpoints, pad_list_to_size
from rasa_nlu.evaluate import (
    plot_confusion_matrix,
    get_evaluation_metrics)
from rasa_nlu.training_data.formats import MarkdownWriter, MarkdownReader

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser for the evaluate script."""

    parser = argparse.ArgumentParser(
            description='evaluates a dialogue model')
    parser.add_argument(
            '-s', '--stories',
            type=str,
            required=True,
            help="file or folder containing stories to evaluate on")
    parser.add_argument(
            '-m', '--max_stories',
            type=int,
            help="maximum number of stories to test on")
    parser.add_argument(
            '-d', '--core',
            required=True,
            type=str,
            help="core model to run with the server")
    parser.add_argument(
            '-u', '--nlu',
            type=str,
            help="nlu model to run with the server. None for regex interpreter")
    parser.add_argument(
            '-o', '--output',
            type=str,
            nargs="?",
            const="story_confmat.pdf",
            help="output path for the created evaluation plot. If not "
                 "specified, no plot will be generated.")
    parser.add_argument(
            '--e2e', '--end-to-end',
            action='store_true',
            help="Run an end-to-end evaluation for combined action and "
                 "intent prediction. Requires a story file in end-to-end "
                 "format.")
    parser.add_argument(
            '--failed',
            type=str,
            default="failed_stories.md",
            help="output path for the failed stories")
    parser.add_argument(
            '--endpoints',
            default=None,
            help="Configuration file for the connectors as a yml file")
    parser.add_argument(
            '--fail_on_prediction_errors',
            action='store_true',
            help="If a prediction error is encountered, an exception "
                 "is thrown. This can be used to validate stories during "
                 "tests, e.g. on travis.")

    utils.add_logging_option_arguments(parser)
    return parser


class EvaluationStore(object):
    """Class storing action, intent and entity predictions and targets."""

    def __init__(
            self,
            action_predictions=None,  # type: Optional[List[str]]
            action_targets=None,  # type: Optional[List[str]]
            intent_predictions=None,  # type: Optional[List[str]]
            intent_targets=None,  # type: Optional[List[str]]
            entity_predictions=None,  # type: Optional[List[Dict[Text, Any]]]
            entity_targets=None  # type: Optional[List[Dict[Text, Any]]]
    ):
        # type: (...) -> None
        self.action_predictions = action_predictions or []
        self.action_targets = action_targets or []
        self.intent_predictions = intent_predictions or []
        self.intent_targets = intent_targets or []
        self.entity_predictions = entity_predictions or []
        self.entity_targets = entity_targets or []

    def add_to_store(
            self,
            action_predictions=None,  # type: Optional[List[str]]
            action_targets=None,  # type: Optional[List[str]]
            intent_predictions=None,  # type: Optional[List[str]]
            intent_targets=None,  # type: Optional[List[str]]
            entity_predictions=None,  # type: Optional[List[Dict[Text, Any]]]
            entity_targets=None  # type: Optional[List[Dict[Text, Any]]]
    ):
        # type: (...) -> None
        """Add items or lists of items to the store"""
        for k, v in locals().items():
            if k != 'self' and v:
                attr = getattr(self, k)
                if isinstance(v, list):
                    attr.extend(v)
                else:
                    attr.append(v)

    def merge_store(self, other):
        # type: (EvaluationStore) -> None
        """Add the contents of other to self"""
        self.add_to_store(action_predictions=other.action_predictions,
                          action_targets=other.action_targets,
                          intent_predictions=other.intent_predictions,
                          intent_targets=other.intent_targets,
                          entity_predictions=other.entity_predictions,
                          entity_targets=other.entity_targets)

    def has_prediction_target_mismatch(self):
        return self.intent_predictions != self.intent_targets or \
               self.entity_predictions != self.entity_targets or \
               self.action_predictions != self.action_targets

    def serialise_targets(self):
        targets = self.action_targets + \
                  self.intent_targets + \
                  self.entity_targets
        return [json.dumps(t) if isinstance(t, dict) else t for t in targets]

    def serialise_predictions(self):
        predictions = self.action_predictions + \
                      self.intent_predictions + \
                      self.entity_predictions

        return [json.dumps(p) if isinstance(p, dict) else p
                for p in predictions]


class WronglyPredictedAction(ActionExecuted):
    """The model predicted the wrong action.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_action"

    def __init__(self, correct_action, predicted_action, timestamp=None):
        self.correct_action = correct_action
        self.predicted_action = predicted_action
        super(WronglyPredictedAction, self).__init__(correct_action,
                                                     timestamp=timestamp)

    def as_story_string(self):
        return "{}   <!-- predicted: {} -->".format(self.correct_action,
                                                    self.predicted_action)


class WronglyClassifiedUserUtterance(UserUttered):
    """The NLU model predicted the wrong user utterance.

    Mostly used to mark wrong predictions and be able to
    dump them as stories."""

    type_name = "wrong_utterance"

    def __init__(self,
                 text,
                 correct_intent,
                 predicted_intent,
                 correct_entities=None,
                 predicted_entities=None,
                 timestamp=None):
        self.text = text
        self.correct_intent = correct_intent
        self.predicted_intent = predicted_intent
        self.correct_entities = correct_entities
        self.predicted_entities = predicted_entities
        super(WronglyClassifiedUserUtterance, self).__init__(
                text, {"name": self.correct_intent}, timestamp=timestamp)

    def _deserialise_entities(self, entities):
        if isinstance(entities, str):
            entities = json.loads(entities)

        return [e for e in entities if isinstance(e, dict)]

    def _md_format_message(self, text, intent, entities):
        message_from_md = MarkdownReader()._parse_training_example(text)
        deserialised_entities = self._deserialise_entities(entities)
        return MarkdownWriter()._generate_message_md(
                {"text": message_from_md.text,
                 "intent": intent,
                 "entities": deserialised_entities}
        )

    def as_story_string(self):
        correct_message = self._md_format_message(self.text,
                                                  self.correct_intent,
                                                  self.correct_entities)
        predicted_message = self._md_format_message(self.text,
                                                    self.predicted_intent,
                                                    self.predicted_entities)
        return ("{}: {}   <!-- predicted: {}: {} -->"
                "").format(self.correct_intent,
                           correct_message,
                           self.predicted_intent,
                           predicted_message)


def _generate_trackers(resource_name, agent, max_stories=None, use_e2e=False):
    story_graph = training.extract_story_graph(resource_name, agent.domain,
                                               agent.interpreter, use_e2e)
    g = TrainingDataGenerator(story_graph, agent.domain,
                              use_story_concatenation=False,
                              augmentation_factor=0,
                              tracker_limit=max_stories)
    return g.generate()


def _clean_entity_results(entity_results):
    return [{k: r[k] for k in ("start", "end", "entity", "value") if k in r}
            for r in entity_results]


def _collect_user_uttered_predictions(event,
                                      partial_tracker,
                                      fail_on_prediction_errors):
    user_uttered_eval_store = EvaluationStore()

    intent_gold = event.parse_data.get("true_intent")
    predicted_intent = event.parse_data.get("intent").get("name")
    if predicted_intent is None:
        predicted_intent = "None"
    user_uttered_eval_store.add_to_store(intent_predictions=predicted_intent,
                                         intent_targets=intent_gold)

    entity_gold = event.parse_data.get("true_entities")
    predicted_entities = event.parse_data.get("entities")

    if entity_gold or predicted_entities:
        if len(entity_gold) > len(predicted_entities):
            predicted_entities = pad_list_to_size(predicted_entities,
                                                  len(entity_gold),
                                                  "None")
        elif len(predicted_entities) > len(entity_gold):
            entity_gold = pad_list_to_size(entity_gold,
                                           len(predicted_entities),
                                           "None")

        user_uttered_eval_store.add_to_store(
                entity_targets=_clean_entity_results(entity_gold),
                entity_predictions=_clean_entity_results(predicted_entities)
        )

    if user_uttered_eval_store.has_prediction_target_mismatch():
        partial_tracker.update(
                WronglyClassifiedUserUtterance(
                        event.text, intent_gold, predicted_intent,
                        user_uttered_eval_store.entity_targets,
                        user_uttered_eval_store.entity_predictions)
        )
        if fail_on_prediction_errors:
            raise ValueError(
                    "NLU model predicted a wrong intent. Failed Story:"
                    " \n\n{}".format(partial_tracker.export_stories()))
    else:
        partial_tracker.update(event)

    return user_uttered_eval_store


def _collect_action_executed_predictions(processor, partial_tracker, event,
                                         fail_on_prediction_errors):
    action_executed_eval_store = EvaluationStore()

    action, _, _ = processor.predict_next_action(partial_tracker)

    predicted = action.name()
    gold = event.action_name

    action_executed_eval_store.add_to_store(action_predictions=predicted,
                                            action_targets=gold)

    if action_executed_eval_store.has_prediction_target_mismatch():
        partial_tracker.update(WronglyPredictedAction(gold, predicted))
        if fail_on_prediction_errors:
            raise ValueError(
                    "Model predicted a wrong action. Failed Story: "
                    "\n\n{}".format(partial_tracker.export_stories()))
    else:
        partial_tracker.update(event)

    return action_executed_eval_store


def _predict_tracker_actions(tracker, agent, fail_on_prediction_errors=False,
                             use_e2e=False):
    processor = agent.create_processor()

    tracker_eval_store = EvaluationStore()

    events = list(tracker.events)

    partial_tracker = DialogueStateTracker.from_events(tracker.sender_id,
                                                       events[:1],
                                                       agent.domain.slots)

    for event in events[1:]:
        if isinstance(event, ActionExecuted):
            action_executed_result = \
                _collect_action_executed_predictions(
                        processor, partial_tracker, event,
                        fail_on_prediction_errors
                )
            tracker_eval_store.merge_store(action_executed_result)
        elif use_e2e and isinstance(event, UserUttered):
            user_uttered_result = \
                _collect_user_uttered_predictions(
                        event, partial_tracker, fail_on_prediction_errors)

            tracker_eval_store.merge_store(user_uttered_result)
        else:
            partial_tracker.update(event)

    return tracker_eval_store, partial_tracker


def collect_story_predictions(
        completed_trackers,  # type: List[DialogueStateTracker]
        agent,  # type: Agent
        fail_on_prediction_errors=False,  # type: bool
        use_e2e=False  # type: bool
):
    # type: (...) -> Tuple[EvaluationStore, List[DialogueStateTracker]]
    """Test the stories from a file, running them through the stored model."""

    story_eval_store = EvaluationStore()
    failed = []
    correct_dialogues = []

    logger.info("Evaluating {} stories\n"
                "Progress:".format(len(completed_trackers)))

    for tracker in tqdm(completed_trackers):
        tracker_results, predicted_tracker = \
            _predict_tracker_actions(tracker, agent,
                                     fail_on_prediction_errors, use_e2e)

        story_eval_store.merge_store(tracker_results)

        if tracker_results.has_prediction_target_mismatch():
            # there is at least one wrong prediction
            failed.append(predicted_tracker)
            correct_dialogues.append(0)
        else:
            correct_dialogues.append(1)

    logger.info("Finished collecting predictions.")
    report, precision, f1, accuracy = get_evaluation_metrics(
            [1] * len(completed_trackers), correct_dialogues)

    log_evaluation_table([1] * len(completed_trackers),
                         "END-TO-END" if use_e2e else "CONVERSATION",
                         report, precision, f1, accuracy,
                         include_report=False)

    return story_eval_store, failed


def log_failed_stories(failed, failed_output):
    """Takes stories as a list of dicts"""

    if not failed_output:
        return

    with io.open(failed_output, 'w', encoding="utf-8") as f:
        if len(failed) == 0:
            f.write("<!-- All stories passed -->")
        else:
            for failure in failed:
                f.write(failure.export_stories())
                f.write("\n\n")


def run_story_evaluation(resource_name, agent,
                         max_stories=None,
                         out_file_stories=None,
                         out_file_plot=None,
                         fail_on_prediction_errors=False,
                         use_e2e=False):
    """Run the evaluation of the stories, optionally plots the results."""

    completed_trackers = _generate_trackers(resource_name, agent,
                                            max_stories, use_e2e)

    story_results, failed = collect_story_predictions(
            completed_trackers, agent, fail_on_prediction_errors, use_e2e)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        report, precision, f1, accuracy = get_evaluation_metrics(
                story_results.serialise_targets(),
                story_results.serialise_predictions()
        )

    if out_file_plot:
        plot_story_evaluation(story_results.action_targets,
                              story_results.action_predictions,
                              report, precision, f1, accuracy, out_file_plot)

    log_failed_stories(failed, out_file_stories)

    return {
        "story_evaluation": {
            "report": report,
            "precision": precision,
            "f1": f1,
            "accuracy": accuracy
        }
    }


def log_evaluation_table(golds, name,
                         report, precision, f1, accuracy,
                         include_report=True):  # pragma: no cover
    """Log the sklearn evaluation metrics."""
    logger.info("Evaluation Results on {} level:".format(name))
    logger.info("\tCorrect:   {} / {}".format(int(len(golds) * accuracy),
                                              len(golds)))
    logger.info("\tF1-Score:  {:.3f}".format(f1))
    logger.info("\tPrecision: {:.3f}".format(precision))
    logger.info("\tAccuracy:  {:.3f}".format(accuracy))

    if include_report:
        logger.info("\tClassification report: \n{}".format(report))


def plot_story_evaluation(test_y, predictions,
                          report, precision, f1, accuracy,
                          out_file):
    """Plot the results of story evaluation"""
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt

    log_evaluation_table(test_y, "ACTION",
                         report, precision, f1, accuracy,
                         include_report=True)

    cnf_matrix = confusion_matrix(test_y, predictions)

    plot_confusion_matrix(cnf_matrix,
                          classes=unique_labels(test_y, predictions),
                          title='Action Confusion matrix')

    fig = plt.gcf()
    fig.set_size_inches(int(20), int(20))
    fig.savefig(out_file, bbox_inches='tight')


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.basicConfig(level=cmdline_args.loglevel)
    _endpoints = AvailableEndpoints.read_endpoints(cmdline_args.endpoints)

    _interpreter = NaturalLanguageInterpreter.create(cmdline_args.nlu,
                                                     _endpoints.nlu)

    _agent = Agent.load(cmdline_args.core,
                        interpreter=_interpreter)

    run_story_evaluation(cmdline_args.stories,
                         _agent,
                         cmdline_args.max_stories,
                         cmdline_args.failed,
                         cmdline_args.output,
                         cmdline_args.fail_on_prediction_errors,
                         cmdline_args.e2e)

    logger.info("Finished evaluation")
