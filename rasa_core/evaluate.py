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

from rasa_core import training
from rasa_core import utils
from rasa_core.agent import Agent
from rasa_core.events import ActionExecuted, UserUttered
from rasa_core.interpreter import RegexInterpreter, RasaNLUInterpreter
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

    utils.add_logging_option_arguments(parser)
    return parser


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


def collect_story_predictions(resource_name, policy_model_path, nlu_model_path,
                              max_stories):
    """Test the stories from a file, running them through the stored model."""

    if nlu_model_path is not None:
        interpreter = RasaNLUInterpreter(model_directory=nlu_model_path)
    else:
        interpreter = RegexInterpreter()

    agent = Agent.load(policy_model_path, interpreter=interpreter)
    story_graph = training.extract_story_graph(resource_name, agent.domain,
                                               interpreter)
    preds = []
    actual = []

    g = TrainingDataGenerator(story_graph, agent.domain,
                              use_story_concatenation=False,
                              tracker_limit=max_stories)
    completed_trackers = g.generate()

    failed_stories = []

    logger.info("Evaluating {} stories\nProgress:"
                "".format(len(completed_trackers)))

    for tracker in tqdm(completed_trackers):
        sender_id = "default-" + uuid.uuid4().hex
        story = {"predicted": [], "actual": []}
        events = list(tracker.events)
        actions_between_utterances = []
        last_prediction = []

        for i, event in enumerate(events[1:]):
            if isinstance(event, UserUttered):
                p, a = align_lists(last_prediction, actions_between_utterances)
                story["predicted"].extend(p)
                story["actual"].extend(a)
                actions_between_utterances = []
                agent.handle_message(event.text, sender_id=sender_id)
                tracker = agent.tracker_store.retrieve(sender_id)
                last_prediction = actions_since_last_utterance(tracker)

            elif isinstance(event, ActionExecuted):
                actions_between_utterances.append(event.action_name)

        if last_prediction:

            preds.extend(last_prediction)
            preds_padding = (len(actions_between_utterances) -
                             len(last_prediction))

            story["predicted"].extend(["None"] * preds_padding)
            preds.extend(story["predicted"])

            actual.extend(actions_between_utterances)
            actual_padding = (len(last_prediction) -
                              len(actions_between_utterances))

            story["actual"].extend(["None"] * actual_padding)
            actual.extend(story["actual"])

        if story["predicted"] != story["actual"]:
            failed_stories.append(story)

    return actual, preds, failed_stories


def log_failed_stories(failed_stories, failed_output):
    """Takes stories as a list of dicts"""

    if not failed_output:
        return

    with io.open(failed_output, 'w') as f:
        if len(failed_stories) == 0:
            f.write("All stories passed")
        else:
            for i, story in enumerate(failed_stories):
                f.write("\n## failed story {}\n".format(i))
                for (p, a) in zip(story["predicted"], story["actual"]):
                    if p == a:
                        f.write("{:40}\n".format(a))
                    else:
                        f.write("{:40} predicted: {:40}\n".format(a, p))


def run_story_evaluation(resource_name, policy_model_path,
                         nlu_model_path=None,
                         max_stories=None,
                         out_file_stories=None,
                         out_file_plot=None):
    """Run the evaluation of the stories, optionally plots the results."""
    test_y, preds, failed_stories = collect_story_predictions(resource_name,
                                                              policy_model_path,
                                                              nlu_model_path,
                                                              max_stories)
    if out_file_plot:
        plot_story_evaluation(test_y, preds, out_file_plot)

    log_failed_stories(failed_stories, out_file_stories)


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
    run_story_evaluation(cmdline_args.stories,
                         cmdline_args.core,
                         cmdline_args.nlu,
                         cmdline_args.max_stories,
                         cmdline_args.failed,
                         cmdline_args.output)
    logger.info("Finished evaluation")
