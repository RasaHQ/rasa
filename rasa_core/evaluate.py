from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import str

import argparse
import io
import logging
import uuid
from difflib import SequenceMatcher
from tqdm import tqdm
from typing import Text, List, Tuple

from rasa_core import training, run
from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.agent import Agent
from rasa_core.events import ActionExecuted, UserUttered, Event
from rasa_core.interpreter import RegexInterpreter, RasaNLUInterpreter
from rasa_core.processor import MessageProcessor
from rasa_core.trackers import DialogueStateTracker
from rasa_core.training.generator import TrainingDataGenerator
from rasa_nlu.evaluate import plot_confusion_matrix, log_evaluation_table

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
            default=100,
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
            default="story_confmat.pdf",
            help="output path for the created evaluation plot. If set to None"
                 "or an empty string, no plot will be generated.")
    parser.add_argument(
            '--failed',
            type=str,
            default="failed_stories.txt",
            help="output path for the failed stories")
    parser.add_argument(
            '--endpoints',
            default=None,
            help="Configuration file for the connectors as a yml file")

    utils.add_logging_option_arguments(parser)
    return parser


# noinspection PyProtectedMember
class WronglyPredictedAction(Event):
    """The model predicted the wrong action."""

    type_name = "wrong_action"

    def __init__(self, correct_action, predicted_action, timestamp=None):
        self.correct_action = correct_action
        self.predicted_action = predicted_action
        super(WronglyPredictedAction, self).__init__(timestamp)

    def as_story_string(self):
        return self.correct_action + \
               "   <!-- predicted: " + self.predicted_action + "-->"


def align_lists(pred, actual):
    # type: (List[Text], List[Text]) -> Tuple[List[Text], List[Text]]
    """Align two lists trying to keep same elements at the same index.

    If lists contain different items at some indices, the algorithm will
    try to find the best alignment and pad with `None`
    values where necessary."""

    padded_pred = []
    padded_actual = []
    s = SequenceMatcher(None, pred, actual)

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        padded_pred.extend(pred[i1:i2])
        padded_pred.extend(["None"] * ((j2 - j1) - (i2 - i1)))

        padded_actual.extend(actual[j1:j2])
        padded_actual.extend(["None"] * ((i2 - i1) - (j2 - j1)))

    return padded_pred, padded_actual


def actions_since_last_utterance(tracker):
    # type: (DialogueStateTracker) -> List[Text]
    """Extract all events after the most recent utterance from the user."""

    actions = []
    for e in reversed(tracker.events):
        if isinstance(e, UserUttered):
            break
        elif isinstance(e, ActionExecuted):
            actions.append(e.action_name)
    actions.reverse()
    return actions


def _generate_trackers(resource_name, agent, max_stories):
    story_graph = training.extract_story_graph(resource_name, agent.domain,
                                               agent.interpreter)
    g = TrainingDataGenerator(story_graph, agent.domain,
                              use_story_concatenation=False,
                              tracker_limit=max_stories)
    return g.generate()


def collect_story_predictions(resource_name,
                              policy_model_path,
                              nlu_model_path,
                              max_stories,
                              endpoints):
    """Test the stories from a file, running them through the stored model."""

    interpreter = RasaNLUInterpreter.create(nlu_model_path, endpoints.nlu)

    agent = Agent.load(policy_model_path,
                       interpreter=interpreter)

    completed_trackers = _generate_trackers(resource_name, agent, max_stories)

    processor = agent.create_processor()  # type: MessageProcessor

    preds = []
    actual = []

    failed = []

    logger.info("Evaluating {} stories\nProgress:"
                "".format(len(completed_trackers)))

    for tracker in tqdm(completed_trackers):
        contains_wrong_prediction = False
        predicted_events = []
        events = list(tracker.events)

        for i, event in enumerate(events[1:]):
            if isinstance(event, ActionExecuted):
                sender_id = "default-" + uuid.uuid4().hex
                partial_tracker = DialogueStateTracker.from_events(
                        sender_id, events[:i + 1], agent.domain.slots)
                action, _, _ = processor.predict_next_action(
                        partial_tracker)

                preds.append(action.name())
                actual.append(event.action_name)

                if action.name() != event.action_name:
                    contains_wrong_prediction = True
                    predicted_events.append(WronglyPredictedAction(
                            event.action_name,
                            action.name()))
                else:
                    predicted_events.append(event)
            else:
                predicted_events.append(event)

        if contains_wrong_prediction:
            failure = DialogueStateTracker.from_events("test",
                                                       predicted_events,
                                                       agent.domain.slots)
            failed.append(failure)

    return actual, preds, failed


def log_failed_stories(failed, failed_output):
    """Takes stories as a list of dicts"""

    if not failed_output:
        return

    with io.open(failed_output, 'w') as f:
        if len(failed) == 0:
            f.write("<!-- All stories passed -->")
        else:
            for failure in failed:
                f.write(failure.export_stories())
                f.write("\n\n")


def run_story_evaluation(resource_name, policy_model_path, endpoints,
                         nlu_model_path=None,
                         max_stories=None,
                         out_file_stories=None,
                         out_file_plot=None):
    """Run the evaluation of the stories, optionally plots the results."""
    test_y, preds, failed = collect_story_predictions(resource_name,
                                                      policy_model_path,
                                                      nlu_model_path,
                                                      max_stories,
                                                      endpoints)
    if out_file_plot:
        plot_story_evaluation(test_y, preds, out_file_plot)

    log_failed_stories(failed, out_file_stories)


def plot_story_evaluation(test_y, preds, out_file):
    """Plot the results. of story evaluation"""
    from sklearn.metrics import confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt

    log_evaluation_table(test_y, preds)
    cnf_matrix = confusion_matrix(test_y, preds)
    plot_confusion_matrix(cnf_matrix, classes=unique_labels(test_y, preds),
                          title='Action Confusion matrix')

    fig = plt.gcf()
    fig.set_size_inches(int(20), int(20))
    fig.savefig(out_file, bbox_inches='tight')


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    logging.basicConfig(level=cmdline_args.loglevel)
    _endpoints = run.read_endpoints(cmdline_args.endpoints)

    run_story_evaluation(cmdline_args.stories,
                         cmdline_args.core,
                         _endpoints,
                         cmdline_args.nlu,
                         cmdline_args.max_stories,
                         cmdline_args.failed,
                         cmdline_args.output)

    logger.info("Finished evaluation")
