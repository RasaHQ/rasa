from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import logging
import os

import numpy as np
import typing
from builtins import range, str
from typing import Optional, Any, List

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.events import ActionExecuted
from rasa_core.events import UserUtteranceReverted, StoryExported
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.ensemble import PolicyEnsemble

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker
    from rasa_core.interpreter import NaturalLanguageInterpreter
    from rasa_core.channels import InputChannel

DEFAULT_FILE_EXPORT_PATH = "stories.md"


class TrainingFinishedException(Exception):
    """Signal a finished online learning. Needed to break out of loops."""
    pass


class OnlinePolicyEnsemble(PolicyEnsemble):
    def __init__(self,
                 base_ensemble,  # type: PolicyEnsemble
                 training_trackers,  # type: List[DialogueStateTracker]
                 max_visual_history=3,  # type: int
                 use_visualization=False  # type: bool
                 ):
        super(OnlinePolicyEnsemble, self).__init__(base_ensemble.policies)

        self.base_ensemble = base_ensemble
        self.training_trackers = training_trackers
        self.max_visual_history = max_visual_history
        self.use_visualization = use_visualization

        self.current_id = 0
        self.extra_intent_examples = []
        self.stories = []

        self.batch_size = 5
        self.epochs = 50

    def run_online_training(self,
                            domain,  # type: Domain
                            interpreter,  # type: NaturalLanguageInterpreter
                            input_channel=None  # type: Optional[InputChannel]
                            ):
        # type: (...) -> None
        from rasa_core.agent import Agent
        if interpreter is None:
            interpreter = RegexInterpreter()

        bot = Agent(domain, self,
                    interpreter=interpreter)
        bot.toggle_memoization(False)

        try:
            # TODO: TB - handle channel is gone!
            bot.handle_channel(input_channel)
        except TrainingFinishedException:
            pass  # training has finished

    def probabilities_using_best_policy(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        # given a state, predict next action via asking a human

        probabilities = self.base_ensemble.probabilities_using_best_policy(
                tracker, domain)
        pred_out = int(np.argmax(probabilities))
        latest_action_was_listen = self._print_history(tracker)

        action_name = domain.action_for_index(pred_out).name()
        colored_name = utils.wrap_with_color(action_name,
                                             utils.bcolors.OKBLUE)
        if latest_action_was_listen:
            print("The bot wants to [{}] due to the intent. "
                  "Is this correct?\n".format(colored_name))

            user_input = utils.request_input(
                    ["1", "2", "3", "0"],
                    "\t1.\tYes\n" +
                    "\t2.\tNo, intent is right but the action is wrong\n" +
                    "\t3.\tThe intent is wrong\n" +
                    "\t0.\tExport current conversations as stories and quit\n")
        else:
            print("The bot wants to [{}]. "
                  "Is this correct?\n".format(colored_name))
            user_input = utils.request_input(
                    ["1", "2", "0"],
                    "\t1.\tYes.\n" +
                    "\t2.\tNo, the action is wrong.\n" +
                    "\t0.\tExport current conversations as stories and quit\n")

        if user_input == "1":
            # max prob prediction was correct
            if action_name == ACTION_LISTEN_NAME:
                print("Next user input:")
            return probabilities

        elif user_input == "2":
            # max prob prediction was false, new action required
            # action wrong
            y = self._request_action(probabilities, domain, tracker)

            # update tracker with new action
            new_action_name = domain.action_for_index(y).name()

            # need to copy tracker, because the tracker will be
            # updated with the new event somewhere else
            training_tracker = tracker.copy()
            training_tracker.update(ActionExecuted(new_action_name))

            self._fit_example(training_tracker, domain)

            self.write_out_story(tracker)

            return utils.one_hot(y, domain.num_actions)

        elif user_input == "3":
            # intent wrong and maybe action wrong
            intent = self._request_intent(tracker, domain)
            latest_message = copy.copy(tracker.latest_message)
            latest_message.intent = intent
            tracker.update(UserUtteranceReverted())
            tracker.update(latest_message)
            for e in domain.slots_for_entities(latest_message.entities):
                tracker.update(e)
            return self.probabilities_using_best_policy(tracker, domain)

        elif user_input == "0":
            self._export_stories(tracker)
            raise TrainingFinishedException()

        else:
            raise Exception(
                    "Incorrect user input received '{}'".format(user_input))

    @staticmethod
    def _export_stories(tracker):
        # export current stories and quit
        file_prompt = ("File to export to (if file exists, this "
                       "will append the stories) "
                       "[{}]: ").format(DEFAULT_FILE_EXPORT_PATH)
        export_file_path = utils.request_input(prompt=file_prompt)

        if not export_file_path:
            export_file_path = DEFAULT_FILE_EXPORT_PATH

        exported = StoryExported(export_file_path)
        tracker.update(exported)
        logger.info("Stories got exported to '{}'.".format(
                os.path.abspath(exported.path)))

    def _fit_example(self, tracker, domain):
        # takes the new example labelled and learns it
        # via taking `epochs` samples of n_batch-1 parts of the training data,
        # inserting our new example and learning them. this means that we can
        # ask the network to fit the example without overemphasising
        # its importance (and therefore throwing off the biases)

        self.training_trackers.append(tracker)
        self.continue_training(self.training_trackers, domain,
                               batch_size=self.batch_size,
                               epochs=self.epochs)

    def write_out_story(self, tracker):
        # takes our new example and writes it in markup story format
        self.stories.append(tracker.export_stories())
