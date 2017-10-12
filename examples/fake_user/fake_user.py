from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from rasa_core.events import Restarted

from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.channels import UserMessage
from rasa_core.channels.channel import InputChannel
from rasa_core.channels.console import ConsoleOutputChannel

logger = logging.getLogger(__name__)


class FakeUserInputChannel(InputChannel):
    """Input channel that reads the user messages from the command line."""

    def __init__(self, tracker_store):
        self.tracker_store = tracker_store
        self.customer = Customer()

    def _record_messages(self, on_message, max_message_limit=None):
        logger.info("Bot loaded. Fake user will automatically respond!")
        num_messages = 0

        while max_message_limit is None or num_messages < max_message_limit:
            tracker = self.tracker_store.retrieve('default')
            text = self.customer.respond_to_action(tracker)
            on_message(UserMessage(text, ConsoleOutputChannel()))
            num_messages += 1

    def start_async_listening(self, message_queue):
        self._record_messages(message_queue.enqueue)

    def start_sync_listening(self, message_handler):
        self._record_messages(message_handler)


class Customer:
    def __init__(self, indecisiveness=0.01, informativeness=1.1):
        self.cuisines = ['french', 'indian', 'chinese', 'sushi', 'british']
        self.amount_of_people = ['2', '3', '5', '6']
        self.city = ['london', 'paris', 'berlin', 'san francisco']
        self.price = ['moderate', 'expensive', 'cheap']
        self.happy = False
        self.complete = False
        self.indecisiveness = indecisiveness
        self.informativeness = informativeness
        self.preferences = {
            'cuisine': np.random.choice(self.cuisines),
            'people': np.random.choice(self.amount_of_people),
            'location': np.random.choice(self.city),
            'price': np.random.choice(self.price)
        }
        init_size = np.random.poisson(informativeness - 1) + 1
        initial_information = np.random.choice(list(self.preferences.keys()),
                                               size=init_size, replace=False)
        entity_strings = ','.join(['{}={}'.format(key, self.preferences[key])
                                   for key in initial_information])

        self.opening_inform = '_inform[{}]'.format(entity_strings)

        self.response_dict = {
            'action_ask_location': '.location',
            'action_ask_numpeople': '.people',
            'action_ask_price': '.price',
            'action_ask_cuisine': '.cuisine',
            'action_ack_findalternatives': '_deny',
            'action_ack_makereservation': 'happy_test',
            'action_ask_moreupdates': '_deny',
            'action_ask_helpmore': 'reshuf',
            'action_on_it': '_deny',
            'action_search_restaurants': '_deny',
            'action_store_slot': '_deny',
            'action_suggest': 'compare',
            'action_greet': self.opening_inform,
            'action_ask_howcanhelp': self.opening_inform,
            'action_goodbye': 'reset'
        }

    def _slot_values(self, tracker):
        values = {}
        for k, s in tracker.slots.items():
            values[k] = s.value
        return values

    def respond_to_action(self, tracker):
        if tracker is None:
            return '_greet'
        else:
            previous_actions = tracker.latest_action_ids
            if (len(previous_actions) < 2 or
                    previous_actions[-1] != ACTION_LISTEN_NAME or
                    previous_actions[-2] not in self.response_dict):
                # if the robot is not listening
                return 0
            output = self.response_dict[previous_actions[-2]]
            if self.complete is True:
                return '_goodbye'
            elif output.startswith('_'):
                return output
            elif output.startswith('.'):
                entity_string = '{}={}'.format(output[1:],
                                               self.preferences[output[1:]])
                return '_inform[{}]'.format(entity_string)
            elif output == 'coinflip':
                response = int(np.random.binomial(1, 0.5))
                return ['_affirm', '_deny'][response]
            elif output == 'reshuf':
                self.rethink()
            elif output == 'compare':
                if self._slot_values(tracker) == self.preferences:
                    self.happy = True
                    return '_affirm'
                else:
                    set_guess = set([k
                                     for k, s in tracker.slots.items()
                                     if s.value])
                    set_true = set(self.preferences.items())
                    fixes = set_true - set_guess
                    print(fixes)
                    entity_string = ','.join(['{}={}'.format(
                            fix[0], fix[1]) for fix in fixes])
                    return '_inform[{}]'.format(entity_string)
            elif output == 'happy_test':
                if self.happy is True:
                    self.complete = True
                    return '_affirm'
                else:
                    return '_deny'
            elif output == 'reset':
                self.__init__(indecisiveness=self.indecisiveness,
                              informativeness=self.informativeness)
                tracker.update(Restarted())
                return '_greet'
        self.rethink()

    def rethink(self):
        if np.random.rand() < self.indecisiveness:
            new_preferences = {
                'cuisine': np.random.choice(self.cuisines),
                'people': np.random.choice(self.amount_of_people),
                'location': np.random.choice(self.city),
                'price': np.random.choice(self.price)
            }
            random_key = np.random.choice(new_preferences)
            self.preferences[random_key] = new_preferences[random_key]
            return '_inform[{}:{}]'.format(
                    random_key, new_preferences[random_key])
        else:
            return '_deny'
