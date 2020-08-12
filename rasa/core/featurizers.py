import io
import jsonpickle
import logging
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Any, Union
import copy
from collections import deque
import scipy.sparse

import rasa.utils.io
from rasa.core import utils
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX, Domain, STATE
from rasa.core.events import ActionExecuted, UserUttered, Form, SlotSet
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.data import DialogueTrainingData
from rasa.utils.common import is_logging_disabled
from rasa.core.interpreter import (
    NaturalLanguageInterpreter,
    RegexInterpreter,
    RasaNLUInterpreter,
)
from rasa.core.constants import USER, PREVIOUS_ACTION, FORM, SLOTS
from rasa.nlu.constants import (
    TEXT,
    INTENT,
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)
from rasa.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class SingleStateFeaturizer:
    """Base class for mechanisms to transform the conversations state into ML formats.

    Subclasses of SingleStateFeaturizer decide how the bot will transform
    the conversation state to a format which a classifier can read:
    feature vector.
    """

    def prepare_from_domain(self, domain: Domain) -> None:
        """Helper method to init based on domain."""

        pass

    def encode(
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> np.ndarray:
        """Encode user input."""

        raise NotImplementedError(
            "SingleStateFeaturizer must have "
            "the capacity to "
            "encode states to a feature vector"
        )

    @staticmethod
    def action_as_index(action: Text, domain: Domain) -> Optional[int]:
        """Encode system action as one-hot vector."""

        if action is None:
            return -1

        return domain.index_for_action(action)

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain encoded in rows."""

        raise NotImplementedError("Featurizer must implement encoding actions.")


class BinarySingleStateFeaturizer(SingleStateFeaturizer):
    """Assumes all features are binary.

    All features should be either on or off, denoting them with 1 or 0.
    """

    def __init__(self) -> None:
        """Declares instant variables."""

        super().__init__()

        self.num_features = None
        self.input_state_map = None

    def prepare_from_domain(self, domain: Domain) -> None:
        """Use Domain to prepare featurizer."""

        self.num_features = domain.num_states
        self.input_state_map = domain.input_state_map

    def encode(
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> np.ndarray:
        """Returns a binary vector indicating which features are active.

        Given a dictionary of states (e.g. 'intent_greet',
        'prev_action_listen',...) return a binary vector indicating which
        features of `self.input_features` are in the bag. NB it's a
        regular double precision float array type.

        For example with two active features out of five possible features
        this would return a vector like `[0 0 1 0 1]`

        If intent features are given with a probability, for example
        with two active features and two uncertain intents out
        of five possible features this would return a vector
        like `[0.3, 0.7, 1.0, 0, 1.0]`.

        If this is just a padding vector we set all values to `-1`.
        padding vectors are specified by a `None` or `[None]`
        value for states.
        """

        if not self.num_features:
            raise Exception(
                "BinarySingleStateFeaturizer was not prepared before encoding."
            )

        if state is None or None in state:
            return np.ones(self.num_features, dtype=np.int32) * -1

        # we are going to use floats and convert to int later if possible
        used_features = np.zeros(self.num_features, dtype=np.float)
        using_only_ints = True
        for state_name, prob in state.items():
            if state_name in self.input_state_map:
                idx = self.input_state_map[state_name]
                used_features[idx] = prob
                using_only_ints = using_only_ints and utils.is_int(prob)
            else:
                logger.debug(
                    "Feature '{}' (value: '{}') could not be found in "
                    "feature map. Make sure you added all intents and "
                    "entities to the domain".format(state_name, prob)
                )

        if using_only_ints:
            # this is an optimization - saves us a bit of memory
            return used_features.astype(np.int32)
        else:
            return used_features

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain encoded in rows as bag of words"""

        return np.eye(domain.num_actions)


class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    """Creates bag-of-words feature vectors.

    User intents and bot action names are split into tokens
    and used to create bag-of-words feature vectors.

    Args:
        split_symbol: The symbol that separates words in
            intets and action names.

        use_shared_vocab: The flag that specifies if to create
            the same vocabulary for user intents and bot actions.
    """

    def __init__(
        self, use_shared_vocab: bool = False, split_symbol: Text = "_"
    ) -> None:
        """inits vocabulary for label bag of words representation"""
        super().__init__()

        self.use_shared_vocab = use_shared_vocab
        self.split_symbol = split_symbol

        self.num_features = None
        self.user_labels = []
        self.slot_labels = []
        self.bot_labels = []

        self.bot_vocab = None
        self.user_vocab = None

    @staticmethod
    def _create_label_token_dict(labels, split_symbol="_") -> Dict[Text, int]:
        """Splits labels into tokens by using provided symbol.

        Creates the lookup dictionary for this tokens.
        Values in this dict are used for featurization.
        """

        distinct_tokens = {
            token for label in labels for token in label.split(split_symbol)
        }
        return {token: idx for idx, token in enumerate(sorted(distinct_tokens))}

    def prepare_from_domain(self, domain: Domain) -> None:
        """Creates internal vocabularies for user intents and bot actions."""

        self.user_labels = domain.intents + domain.entity_states
        self.slot_labels = domain.slot_states + domain.form_names
        self.bot_labels = domain.action_names

        if self.use_shared_vocab:
            self.bot_vocab = self._create_label_token_dict(
                self.bot_labels + self.user_labels, self.split_symbol
            )
            self.user_vocab = self.bot_vocab
        else:
            self.bot_vocab = self._create_label_token_dict(
                self.bot_labels, self.split_symbol
            )
            self.user_vocab = self._create_label_token_dict(
                self.user_labels, self.split_symbol
            )

        self.num_features = (
            len(self.user_vocab) + len(self.slot_labels) + len(self.bot_vocab)
        )

    def encode(
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> np.ndarray:
        """Returns a binary vector indicating which tokens are present."""

        if not self.num_features:
            raise Exception(
                "LabelTokenizerSingleStateFeaturizer "
                "was not prepared before encoding."
            )

        if state is None or None in state:
            return np.ones(self.num_features, dtype=np.int32) * -1

        # we are going to use floats and convert to int later if possible
        used_features = np.zeros(self.num_features, dtype=np.float)
        using_only_ints = True
        for state_name, prob in state.items():
            using_only_ints = using_only_ints and utils.is_int(prob)
            if state_name in self.user_labels:
                if PREV_PREFIX + ACTION_LISTEN_NAME in state:
                    # else we predict next action from bot action and memory
                    for t in state_name.split(self.split_symbol):
                        used_features[self.user_vocab[t]] += prob

            elif state_name in self.slot_labels:
                offset = len(self.user_vocab)
                idx = self.slot_labels.index(state_name)
                used_features[offset + idx] += prob

            elif state_name[len(PREV_PREFIX) :] in self.bot_labels:
                action_name = state_name[len(PREV_PREFIX) :]
                for t in action_name.split(self.split_symbol):
                    offset = len(self.user_vocab) + len(self.slot_labels)
                    idx = self.bot_vocab[t]
                    used_features[offset + idx] += prob

            else:
                logger.warning(
                    f"Feature '{state_name}' could not be found in feature map."
                )

        if using_only_ints:
            # this is an optimization - saves us a bit of memory
            return used_features.astype(np.int32)
        else:
            return used_features

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain encoded in rows as bag of words"""

        encoded_all_actions = np.zeros(
            (domain.num_actions, len(self.bot_vocab)), dtype=np.int32
        )
        for idx, name in enumerate(domain.action_names):
            for t in name.split(self.split_symbol):
                encoded_all_actions[idx, self.bot_vocab[t]] = 1
        return encoded_all_actions


class E2ESingleStateFeaturizer(SingleStateFeaturizer):
    def __init__(self) -> None:

        super().__init__()
        self.output_shapes = {}
        self.slot_names = []
        self.slot_states = []
        self.form_states = []
        self.entities = []
        self.action_names = []
        self.intents = []

    def prepare_from_domain(self, domain: Domain) -> None:
        # storing slot names so that the features are always added in the same order
        self.slot_names = [slot.name for slot in domain.slots]
        self.slot_states = domain.slot_states
        self.form_states = domain.form_names
        self.entities = domain.entities
        self.action_names = domain.action_names
        self.intents = domain.intents

        self.output_shapes[ENTITIES] = len(
            self.slot_states + self.form_states + self.entities
        )

    def fill_in_features(self, features: List) -> List:
        shapes = self.output_shapes

        for feature in features:
            intent_rows_to_fill = np.where(feature[:, 3] != -1)[0]
            user_text_rows_to_fill = np.where(feature[:, 3] != 1)[0]
            action_names_rows_to_fill = np.where(feature[:, 7] != -1)[0]
            action_text_rows_to_fill = np.where(feature[:, 7] != 1)[0]

            for key in shapes.keys():
                if INTENT in key:
                    feature[np.array(intent_rows_to_fill), 2] = [
                        np.ones((1, shapes.get(key))) * -1
                    ] * len(intent_rows_to_fill)
                elif ACTION_NAME in key:
                    feature[np.array(action_names_rows_to_fill), 6] = [
                        np.ones((1, shapes.get(key))) * -1
                    ] * len(action_names_rows_to_fill)
                elif ACTION_TEXT in key:
                    if "sparse" in key:
                        feature[np.array(action_text_rows_to_fill), 4] = [
                            scipy.sparse.coo_matrix((1, shapes.get(key)))
                        ] * len(action_text_rows_to_fill)
                    elif "dense" in key:
                        feature[np.array(action_text_rows_to_fill), 5] = [
                            np.ones((1, shapes.get(key))) * -1
                        ] * len(action_text_rows_to_fill)
                else:
                    if "sparse" in key:
                        feature[np.array(user_text_rows_to_fill), 0] = [
                            scipy.sparse.coo_matrix((1, shapes.get(key)))
                        ] * len(user_text_rows_to_fill)
                    elif "dense" in key:
                        feature[np.array(user_text_rows_to_fill), 1] = [
                            np.ones((1, shapes.get(key))) * -1
                        ] * len(user_text_rows_to_fill)

        return features

    def _get_slot_and_entity_features(self, state: STATE) -> np.ndarray:
        binary_features = np.zeros(
            (len(self.slot_states + self.form_states + self.entities))
        )
        if state.get(SLOTS):
            # collect slot features
            current_slot_names = state.get(SLOTS).keys()
            slot_values = [
                np.array(state.get(SLOTS)[slot_name])
                for slot_name in self.slot_names
                if slot_name in current_slot_names
            ]
            slot_values = np.hstack(slot_values)
            binary_features[: len(self.slot_states)] = slot_values
        if state.get(FORM):
            # featurize forms
            form_values = np.zeros((len(self.form_states)))
            form_values[self.form_states.index(state.get(FORM).get("name"))] += 1
            binary_features[
                len(self.slot_states) : len(self.slot_states + self.form_states)
            ] = form_values
        if state.get(USER):
            if state[USER].get(ENTITIES):
                entities = state[USER].get(ENTITIES)
                for entity in entities:
                    binary_features[
                        len(self.slot_states + self.form_states)
                        + self.entities.index(entity)
                    ] += 1

        return binary_features

    def _check_dense_features(
        self, dense_features: Tuple[np.ndarray, np.ndarray]
    ) -> bool:
        if dense_features[1] is None:
            return False
        if dense_features[1].size == 0:
            return False
        return True

    def _record_shape(
        self,
        attribute: Text,
        sparse_features: Tuple[scipy.sparse.spmatrix, scipy.sparse.spmatrix],
        dense_features: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        if attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            if not sparse_features[1] is None:
                if f"{attribute}_sparse" not in self.output_shapes.keys():
                    self.output_shapes[f"{attribute}_sparse"] = sparse_features[
                        1
                    ].shape[-1]

            if self._check_dense_features(dense_features):
                if f"{attribute}_dense" not in self.output_shapes.keys():
                    self.output_shapes[f"{attribute}_dense"] = dense_features[1].shape[
                        -1
                    ]
        else:
            if not sparse_features[0] is None:
                if f"{attribute}" not in self.output_shapes.keys():
                    self.output_shapes[f"{attribute}"] = sparse_features[0].shape[-1]

    def _construct_message(self, state: Dict[Text, Text], state_comes_from: Text):
        if state_comes_from == USER:
            if state.get(INTENT):
                message = Message(data={INTENT: state.get(INTENT)})
                attribute = INTENT
            else:
                message = Message(state.get(TEXT))
                attribute = TEXT
        else:
            if state.get(ACTION_NAME):
                message = Message(data={ACTION_NAME: state.get(ACTION_NAME)})
                attribute = ACTION_NAME
            else:
                message = Message(data={ACTION_TEXT: state.get(ACTION_TEXT)})
                attribute = ACTION_TEXT
        return message, attribute

    def _extract_features(
        self,
        state: Dict[str, Union[str, Tuple[Union[float, str]]]],
        state_comes_from: Text,
        interpreter: NaturalLanguageInterpreter,
    ) -> List[Union[scipy.sparse.spmatrix, np.ndarray]]:

        message, attribute = self._construct_message(state, state_comes_from)

        parsed_message = interpreter.synchronous_parse_message(message, attribute)
        sparse_features = parsed_message.get_sparse_features(attribute)
        dense_features = parsed_message.get_dense_features(attribute)

        self._record_shape(attribute, sparse_features, dense_features)

        # output_features = [sparse_features_for_text, dense_features_for_text, sparse_features_for_name, 1 for text OR -1 for name]
        if attribute.endswith(TEXT):
            output_features = [sparse_features[1], dense_features[1], None, 1]
        else:
            output_features = [None, None, sparse_features[0].sum(0), -1]
        return output_features

    def _tokenizer_in_pipeline(self, interpreter: RasaNLUInterpreter) -> bool:
        from rasa.nlu.tokenizers.tokenizer import Tokenizer

        tokenizer_in_pipeline = any(
            [
                isinstance(component, Tokenizer)
                for component in interpreter.interpreter.pipeline
            ]
        )
        if tokenizer_in_pipeline:
            return tokenizer_in_pipeline
        else:
            # TODO: do these warnings make sense? or are they confusing and it is better to remove them?
            # logger.warning(
            #     "No tokenizer is included in the NLU pipeline. Features for intents and actions will be featurized on the fly."
            # )
            return tokenizer_in_pipeline

    def _count_featurizer_in_pipeline(self, interpreter: RasaNLUInterpreter) -> bool:
        from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
            CountVectorsFeaturizer,
        )

        count_featurizer_in_pipeline = any(
            [
                isinstance(component, CountVectorsFeaturizer)
                for component in interpreter.interpreter.pipeline
            ]
        )
        if count_featurizer_in_pipeline:
            return count_featurizer_in_pipeline
        else:
            # logger.warning(
            #     "No count vectors featurizer is included in the NLU pipeline. Features for intents and actions will be featurized on the fly."
            # )
            return count_featurizer_in_pipeline

    def _interpreter_is_suitable_for_core_featurization(
        self, interpreter: NaturalLanguageInterpreter
    ) -> bool:
        if isinstance(interpreter, RasaNLUInterpreter):
            return self._tokenizer_in_pipeline(
                interpreter
            ) and self._count_featurizer_in_pipeline(interpreter)
        else:
            # logger.warning(
            #     "No trained NLU model was loaded. Features for intents and actions will be featurized on the fly."
            # )
            return False

    def process_state_without_trained_nlu(self, state: STATE):
        intent_features = np.zeros((1, len(self.intents)))
        action_name_features = np.zeros((1, len(self.action_names)))
        if state.get(USER):
            intent = state.get(USER).get(INTENT)
            if intent:
                intent_features[0, self.intents.index(intent)] += 1
        if state.get(PREVIOUS_ACTION):
            action_name = state.get(PREVIOUS_ACTION).get(ACTION_NAME)
            if action_name:
                action_name_features[0, self.action_names.index(action_name)] += 1
        user_features = [None, None, intent_features, -1]
        action_features = [None, None, action_name_features, -1]
        self.output_shapes[INTENT] = intent_features.shape[-1]
        self.output_shapes[ACTION_NAME] = action_name_features.shape[-1]
        return user_features + action_features

    def encode(
        self, state: STATE, interpreter: NaturalLanguageInterpreter
    ) -> np.ndarray:
        slot_and_entity_features = self._get_slot_and_entity_features(state)
        if state == {}:
            return np.array(
                [None, None, None, 0, None, None, None, 0, slot_and_entity_features]
            )

        if not self._interpreter_is_suitable_for_core_featurization(interpreter):
            return np.array(
                self.process_state_without_trained_nlu(state)
                + [slot_and_entity_features]
            )

        state_extracted_features = {
            key: self._extract_features(state.get(key), key, interpreter)
            for key in [USER, PREVIOUS_ACTION]
            if state.get(key)
        }

        if USER not in state_extracted_features.keys():
            state_extracted_features[USER] = [None, None, None, 0]

        # unify features into a list
        return np.array(
            state_extracted_features[USER]
            + state_extracted_features[PREVIOUS_ACTION]
            + [slot_and_entity_features]
        )

    def _is_action_text(
        self, action: Text, interpreter: NaturalLanguageInterpreter
    ) -> Tuple[bool, Any, Any]:
        """
        Checking whether a given action name from the domain is text or action_name
        """

        # check that there is a featurizer trained for the action name,
        # i.e., that we have encountered action_names in the dataset
        if ACTION_NAME in self.output_shapes.keys():
            action_name_features = self._extract_features(
                {ACTION_NAME: action}, PREVIOUS_ACTION, interpreter
            )
            num_name_elements = np.count_nonzero(action_name_features[2])
        else:
            action_name_features = None
            num_name_elements = -1

        if (
            f"{ACTION_TEXT}_sparse" in self.output_shapes.keys()
            or f"{ACTION_TEXT}_dense" in self.output_shapes.keys()
        ):
            action_text_features = self._extract_features(
                {ACTION_TEXT: action}, PREVIOUS_ACTION, interpreter
            )
            if f"{ACTION_TEXT}_sparse" in self.output_shapes.keys():
                num_text_elements = action_text_features[0].nnz
            elif f"{ACTION_TEXT}_dense" in self.output_shapes.keys():
                num_text_elements = np.count_nonzero(action_text_features[1])
        else:
            action_text_features = None
            num_text_elements = -1

        return (
            num_text_elements > num_name_elements,
            action_name_features,
            action_text_features,
        )

    def create_encoded_all_actions(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ):

        label_data = []
        # if we're doing rasa trin core without trained NLU model
        if not self._interpreter_is_suitable_for_core_featurization(interpreter):
            label_features = np.expand_dims(np.eye(len(domain.action_names)), 1)
            label_data = [
                (j, [None, None, label_features[j]])
                for j, action in enumerate(domain.action_names)
            ]
            return label_data

        for j, action in enumerate(domain.action_names):
            (
                is_action_text,
                action_name_features,
                action_text_features,
            ) = self._is_action_text(action, interpreter)

            if is_action_text:
                action_features = action_text_features
                if self.output_shapes.get(f"{ACTION_NAME}"):
                    action_features[2] = (
                        np.ones((1, self.output_shapes.get(f"{ACTION_NAME}"))) * -1
                    )
            else:
                action_features = action_name_features
                if self.output_shapes.get(f"{ACTION_TEXT}_sparse"):
                    action_features[0] = scipy.sparse.coo_matrix(
                        (1, self.output_shapes.get(f"{ACTION_TEXT}_sparse"))
                    )
                if self.output_shapes.get(f"{ACTION_TEXT}_dense"):
                    action_features[1] = (
                        np.ones((1, self.output_shapes.get(f"{ACTION_TEXT}_dense")))
                        * -1
                    )
            label_data.append((j, action_features))

        return label_data


class TrackerFeaturizer:
    """Base class for actual tracker featurizers."""

    def __init__(
        self, state_featurizer: Optional[SingleStateFeaturizer] = None
    ) -> None:

        self.state_featurizer = state_featurizer

    @staticmethod
    def _unfreeze_states(states: deque) -> List[STATE]:
        return [
            {key: dict(value) for key, value in dict(state).items()} for state in states
        ]

    def _create_states(
        self, tracker: DialogueStateTracker, domain: Domain,
    ) -> List[STATE]:
        """Create states: a list of dictionaries.

        If use_intent_probabilities is False (default behaviour),
        pick the most probable intent out of all provided ones and
        set its probability to 1.0, while all the others to 0.0.
        """

        states = tracker.past_states(domain)

        return self._unfreeze_states(states)

    def _pad_states(self, states: List[Any]) -> List[Any]:
        """Pads states."""

        return states

    def _featurize_states(
        self,
        trackers_as_states: List[List[STATE]],
        interpreter: NaturalLanguageInterpreter,
    ) -> Tuple[np.ndarray, List[int]]:
        """Create X."""

        features = []
        true_lengths = []

        for tracker_states in trackers_as_states:

            # len(trackers_as_states) = 1 means
            # it is called during prediction or we have
            # only one story, so no padding is needed

            story_features = [
                self.state_featurizer.encode(state, interpreter)
                for state in tracker_states
            ]
            dialogue_len = len(story_features)

            if not story_features == []:
                features.append(np.stack(story_features))
                true_lengths.append(dialogue_len)

        if isinstance(self.state_featurizer, E2ESingleStateFeaturizer):
            features = self.state_featurizer.fill_in_features(features)

        # noinspection PyPep8Naming
        X = np.array(features)

        return X, true_lengths

    def _featurize_labels(
        self, trackers_as_actions: List[List[Text]], domain: Domain
    ) -> np.ndarray:
        """Create y."""

        labels = []
        for tracker_actions in trackers_as_actions:

            if len(trackers_as_actions) > 1:
                tracker_actions = self._pad_states(tracker_actions)

            story_labels = [
                self.state_featurizer.action_as_index(action, domain)
                for action in tracker_actions
            ]

            labels.append(story_labels)

        y = np.array(labels)
        if y.ndim == 2 and isinstance(self, MaxHistoryTrackerFeaturizer):
            # if it is MaxHistoryFeaturizer, remove time axis
            y = y.squeeze(-1)

        return y

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[STATE]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions."""

        raise NotImplementedError(
            "Featurizer must have the capacity to encode trackers to feature vectors"
        )

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> DialogueTrainingData:
        """Create training data."""

        if self.state_featurizer is None:
            raise ValueError(
                "Variable 'state_featurizer' is not set. Provide "
                "'SingleStateFeaturizer' class to featurize trackers."
            )

        self.state_featurizer.prepare_from_domain(domain)

        (trackers_as_states, trackers_as_actions) = self.training_states_and_actions(
            trackers, domain
        )

        # noinspection PyPep8Naming
        X, true_lengths = self._featurize_states(trackers_as_states, interpreter)
        y = self._featurize_labels(trackers_as_actions, domain)

        return DialogueTrainingData(X, y, true_lengths)

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[Dict[Text, float]]]:
        """Transforms list of trackers to lists of states for prediction."""

        raise NotImplementedError(
            "Featurizer must have the capacity to create feature vector"
        )

    # noinspection PyPep8Naming
    def create_X(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> np.ndarray:
        """Create X for prediction."""

        trackers_as_states = self.prediction_states(trackers, domain)
        X, _ = self._featurize_states(trackers_as_states, interpreter)
        return X

    def persist(self, path) -> None:
        featurizer_file = os.path.join(path, "featurizer.json")
        rasa.utils.io.create_directory_for_file(featurizer_file)

        # noinspection PyTypeChecker
        rasa.utils.io.write_text_file(str(jsonpickle.encode(self)), featurizer_file)

    @staticmethod
    def load(path) -> Optional["TrackerFeaturizer"]:
        """Loads the featurizer from file."""

        featurizer_file = os.path.join(path, "featurizer.json")
        if os.path.isfile(featurizer_file):
            return jsonpickle.decode(rasa.utils.io.read_file(featurizer_file))
        else:
            logger.error(
                "Couldn't load featurizer for policy. "
                "File '{}' doesn't exist.".format(featurizer_file)
            )
            return None


class FullDialogueTrackerFeaturizer(TrackerFeaturizer):
    """Creates full dialogue training data for time distributed architectures.

    Creates training data that uses each time output for prediction.
    Training data is padded up to the length of the longest dialogue with -1.
    """

    def __init__(self, state_featurizer: SingleStateFeaturizer) -> None:
        super().__init__(state_featurizer)
        self.max_len = None

    @staticmethod
    def _calculate_max_len(trackers_as_actions) -> Optional[int]:
        """Calculate the length of the longest dialogue."""

        if trackers_as_actions:
            return max([len(states) for states in trackers_as_actions])
        else:
            return None

    def _pad_states(self, states: List[Any]) -> List[Any]:
        """Pads states up to max_len."""

        if len(states) < self.max_len:
            states += [None] * (self.max_len - len(states))

        return states

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[STATE]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.

        Training data is padded up to the length of the longest dialogue with -1.
        """

        trackers_as_states = []
        trackers_as_actions = []

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(trackers, desc="Processed trackers", disable=is_logging_disabled())
        for tracker in pbar:
            states = self._create_states(tracker, domain)

            delete_first_state = False
            actions = []
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be
                        # predicted at a stories start
                        actions.append(event.action_name or event.e2e_text)
                    else:
                        # unpredictable actions can be
                        # only the first in the story
                        if delete_first_state:
                            raise Exception(
                                "Found two unpredictable "
                                "actions in one story."
                                "Check your story files."
                            )
                        else:
                            delete_first_state = True

            if delete_first_state:
                states = states[1:]

            trackers_as_states.append(states[:-1])
            trackers_as_actions.append(actions)

        self.max_len = self._calculate_max_len(trackers_as_actions)
        logger.debug(f"The longest dialogue has {self.max_len} actions.")

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[STATE]]:
        """Transforms list of trackers to lists of states for prediction."""

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]

        return trackers_as_states


class MaxHistoryTrackerFeaturizer(TrackerFeaturizer):
    """Slices the tracker history into max_history batches.

    Creates training data that uses last output for prediction.
    Training data is padded up to the max_history with -1.
    """

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
    ) -> None:

        super().__init__(state_featurizer)
        self.max_history = max_history
        self.remove_duplicates = remove_duplicates

    @staticmethod
    def slice_state_history(
        states: List[STATE], slice_length: Optional[int],
    ) -> List[STATE]:
        """Slices states from the trackers history.
        If the slice is at the array borders, padding will be added to ensure
        the slice length.
        """
        slice_end = len(states)
        if slice_length is None:
            slice_start = 0
        else:
            slice_start = max(0, slice_end - slice_length)

        state_features = states[slice_start:]
        return state_features

    @staticmethod
    def _hash_example(
        states: List[STATE], action: Text, tracker: DialogueStateTracker,
    ) -> int:
        """Hash states for efficient deduplication."""
        frozen_states = tuple(
            s if s is None else tracker.freeze_current_state(s) for s in states
        )
        frozen_actions = (action,)
        return hash((frozen_states, frozen_actions))

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[STATE]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.
        Training data is padded up to the max_history with -1.
        """

        trackers_as_states = []
        trackers_as_actions = []

        # from multiple states that create equal featurizations
        # we only need to keep one.
        hashed_examples = set()

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(trackers, desc="Processed trackers", disable=is_logging_disabled())
        for tracker in pbar:
            states = self._create_states(tracker, domain)

            idx = 0
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be
                        # predicted at a stories start
                        sliced_states = self.slice_state_history(
                            states[: idx + 1], self.max_history
                        )
                        if self.remove_duplicates:
                            hashed = self._hash_example(
                                sliced_states,
                                event.action_name or event.e2e_text,
                                tracker,
                            )

                            # only continue with tracker_states that created a
                            # hashed_featurization we haven't observed
                            if hashed not in hashed_examples:
                                hashed_examples.add(hashed)
                                trackers_as_states.append(sliced_states)
                                trackers_as_actions.append(
                                    [event.action_name or event.e2e_text]
                                )
                        else:
                            trackers_as_states.append(sliced_states)
                            trackers_as_actions.append(
                                [event.action_name or event.e2e_text]
                            )

                        pbar.set_postfix(
                            {"# actions": "{:d}".format(len(trackers_as_actions))}
                        )
                    idx += 1

        logger.debug("Created {} action examples.".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[STATE]]:
        """Transforms list of trackers to lists of states for prediction."""

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]
        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]

        return trackers_as_states
