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
from rasa_core.channels.console import ConsoleInputChannel
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
            bot.handle_channel(
                    input_channel if input_channel else ConsoleInputChannel())
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

    def continue_training(self, trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None
        for p in self.policies:
            p.continue_training(trackers, domain, **kwargs)

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

    def _request_intent(self, tracker, domain):
        # take in some argument and ask which intent it should have been
        # save the intent to a json like file
        colored_user_msg = utils.wrap_with_color(tracker.latest_message.text,
                                                 utils.bcolors.OKGREEN)
        print("------\n")
        print("Message:\n")
        print(tracker.latest_message.text)
        print("User said:\t {}".format(colored_user_msg))
        print("What intent is this?\t")
        for idx, intent in enumerate(domain.intents):
            print('\t{}\t{}'.format(idx, intent))
        out = int(utils.request_input(
                utils.str_range_list(0, len(domain.intents))))
        json_example = {
            'text': tracker.latest_message.text,
            'intent': domain.intents[out]
        }
        self.extra_intent_examples.append(json_example)
        intent_name = domain.intents[out]
        return {'name': intent_name, 'confidence': 1.0}

    def _print_history(self, tracker):
        # prints the historical interactions between the bot and the user,
        # to help with correctly identifying the action
        latest_listen_flag = False
        tr_json = []
        for tr in tracker.generate_all_prior_trackers():
            tr_json.append({
                'action': tr.latest_action_name,
                'intent': tr.latest_message.intent[
                    'name'] if tr.latest_message.intent else "",
                'entities': tr.latest_message.entities
            })

        print("------")
        print("Chat history:\n")
        tr_json = tr_json[-self.max_visual_history:]
        n_history = len(tr_json)
        for idx, hist_tracker in enumerate(tr_json):

            print("\tbot did:\t{}\n".format(hist_tracker['action']))
            if hist_tracker['action'] == 'action_listen':
                if idx < n_history - 1:
                    print("\tuser did:\t{}\n".format(hist_tracker['intent']))
                    for entity in hist_tracker['entities']:
                        print("\twith {}:\t{}\n".format(entity['entity'],
                                                        entity['value']))
                if idx == n_history - 1:
                    print("\tuser said:\t{}\n".format(
                            utils.wrap_with_color(tracker.latest_message.text,
                                                  utils.bcolors.OKGREEN)))
                    print("\t\t whose intent is:\t{}\n".format(
                            hist_tracker['intent']))
                    for entity in hist_tracker['entities']:
                        print("\twith {}:\t{}\n".format(entity['entity'],
                                                        entity['value']))
                    latest_listen_flag = True
        slot_strs = []
        for k, s in tracker.slots.items():
            colored_value = utils.wrap_with_color(str(s.value),
                                                  utils.bcolors.WARNING)
            slot_strs.append("{}: {}".format(k, colored_value))
        print("we currently have slots: {}\n".format(", ".join(slot_strs)))

        print("------")
        return latest_listen_flag

    def _request_action(self, predictions, domain, tracker):
        # given the intent and the text (NOT IMPLEMENTED)
        # what is the correct action?
        self._print_history(tracker)
        print("what is the next action for the bot?\n")

        for idx in range(domain.num_actions):
            action_name = domain.action_for_index(idx).name()
            print("{:>10}{:>40}    {:03.2f}".format(idx, action_name,
                                                    predictions[idx]))

        out = int(utils.request_input(
                utils.str_range_list(0, domain.num_actions)))
        print("thanks! The bot will now "
              "[{}]\n -----------".format(domain.action_for_index(out).name()))
        return out
