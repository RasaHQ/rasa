from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import typing

from typing import \
    Any, List, Optional, Text, Dict, Callable

import numpy as np
from copy import deepcopy
from rasa_core.policies import Policy
from rasa_core.featurizers import \
    FullDialogueTrackerFeaturizer, LabelTokenizerSingleStateFeaturizer

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
except ImportError:
    tf = None


class EmbeddingPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = False

    @classmethod
    def _standard_featurizer(cls):
        return FullDialogueTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer())

    config = {
        # nn architecture
        "num_hidden_layers_a": 2,
        "hidden_layer_size_a": [256, 128],
        "num_hidden_layers_b": 0,
        "hidden_layer_size_b": [],
        "batch_size": 16,
        "epochs": 3000,

        # embedding parameters
        "embed_dim": 10,
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        "mu_neg": -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        "similarity_type": 'cosine',  # string 'cosine' or 'inner'
        "num_neg": 20,
        "use_max_sim_neg": True,  # flag which loss function to use

        # regularization
        "C2": 0.001,
        "C_emb": 0.8,

        "droprate_a": 0.1,
        "droprate_b": 0.1,
        "droprate_c": 0.2,
        "droprate_rnn": 0.1,
        "droprate_out": 0.1,
    }

    def _load_nn_architecture_params(self):
        self.num_hidden_layers_a = self.config['num_hidden_layers_a']
        self.hidden_layer_size_a = self.config['hidden_layer_size_a']
        self.num_hidden_layers_b = self.config['num_hidden_layers_b']
        self.hidden_layer_size_b = self.config['hidden_layer_size_b']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']

    def _load_embedding_params(self):
        self.embed_dim = self.config['embed_dim']
        self.mu_pos = self.config['mu_pos']
        self.mu_neg = self.config['mu_neg']
        self.similarity_type = self.config['similarity_type']
        self.num_neg = self.config['num_neg']
        self.use_max_sim_neg = self.config['use_max_sim_neg']

    def _load_regularization_params(self):
        self.C2 = self.config['C2']
        self.C_emb = self.config['C_emb']

        self.droprate_a = self.config['droprate_a']
        self.droprate_b = self.config['droprate_b']
        self.droprate_c = self.config['droprate_c']
        self.droprate_rnn = self.config['droprate_rnn']
        self.droprate_out = self.config['droprate_out']

    def _load_flag_if_tokenize_intents(self):
        self.intent_tokenization_flag = self.config[
                                            'intent_tokenization_flag']
        self.intent_split_symbol = self.config[
                                        'intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(
            self,
            featurizer=None,  # type: Optional[FullDialogueTrackerFeaturizer]
            intent_dict=None,  # type: Optional[Dict[Text, int]]
            intent_token_dict=None,  # type: Optional[Dict[Text, int]]
            session=None,  # type: Optional[tf.Session]
            graph=None,  # type: Optional[tf.Graph]
            intent_placeholder=None,  # type: Optional[tf.Tensor]
            embedding_placeholder=None,  # type: Optional[tf.Tensor]
            similarity_op=None  # type: Optional[tf.Tensor]
     ):
        # type: (...) -> None
        self._check_tensorflow()
        super(EmbeddingPolicy, self).__init__(featurizer)

        # nn architecture parameters
        self._load_nn_architecture_params()
        # embedding parameters
        self._load_embedding_params()
        # regularization
        self._load_regularization_params()
        # flag if tokenize intents
        self._load_flag_if_tokenize_intents()

        self.mean_time = None
        # transform intents to numbers
        self.action_dict = {}  # encode intents with numbers
        self.action_token_dict = {}  # encode words in intents with numbers

        # tf related instances
        self.session = session
        self.graph = graph
        self.intent_placeholder = intent_placeholder
        self.embedding_placeholder = embedding_placeholder
        self.similarity_op = similarity_op

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> None
        """Trains the policy on given training trackers."""

        # dealing with training data
        logger.debug('Started to train embedding policy.')

        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)

        action_token_dict = self.featurizer.state_featurizer.bot_vocab
        split_symbol = self.featurizer.state_featurizer.split_symbol

        # check if number of negatives is less than number of actions
        logger.debug("Check if num_neg {} is smaller "
                     "than number of actions {}, "
                     "else set num_neg to the number of actions - 1"
                     "".format(self.num_neg, domain.num_actions))
        self.num_neg = min(self.num_neg, domain.num_actions - 1)

        # get training data
        prev_start = len(self.featurizer.state_featurizer.user_vocab)
        prev_end = prev_start + len(action_token_dict)

        # do not include prev actions
        X = training_data.X[:, :, :prev_start]
        extras = training_data.X[:, :, prev_end:]

        dialogue_len = X.shape[1]
        self.mean_time = np.mean(training_data.true_length)

        actions_for_X = training_data.y.argmax(axis=-1)

        Y = np.zeros((X.shape[0], dialogue_len, len(action_token_dict)), dtype=int)
        for story_idx, action_ids in enumerate(actions_for_X):
            for time_idx, action_idx in enumerate(action_ids):
                action = domain.action_names[action_idx]
                for t in action.split(split_symbol):
                    Y[story_idx, time_idx, action_token_dict[t]] = 1



    def continue_training(self, trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None
        """Continues training an already trained policy."""

        pass

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
        after seeing the tracker.

        Returns the list of probabilities for the next actions"""

        exit("Policy must have the capacity "
                                  "to predict.")

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to a storage."""
        super(EmbeddingPolicy, self).persist(path)

    @classmethod
    def load(cls, path):
        # type: (Text) -> Policy
        """Loads a policy from the storage.

        Needs to load its featurizer"""

        exit("Policy must have the capacity "
                                  "to load itself.")
