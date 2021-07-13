"""
NOTE:
- it would be great if we could turn the state featurizer into a combination of
  state item featurizers (cf. target featurizers)
-
"""

from abc import abstractmethod
import logging
from rasa.shared.nlu.training_data.features import Features
from typing import List, Optional, Dict, Sequence, Text, Any, Iterable

from rasa.shared.core.domain import SubState, State, Domain
from rasa.shared.core.domain import Domain
from rasa.shared.core import state as state_utils
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import (
    ENTITIES,
    ACTION_NAME,
    INTENT,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    USER,
    SLOTS,
)

logger = logging.getLogger(__name__)

from rasa.core.featurizers import substate_featurizer
from rasa.core.featurizers.substate_featurizer import (
    FallbackSubStateFeaturizer,
    SetupMixin,
    SubStateFeaturizerUsingInterpreter,
)


class StateFeaturizer(SetupMixin):
    """Base class to transform the dialogue state into an ML format.

    Subclasses of SingleStateFeaturizer will decide how a bot will
    transform the dialogue state into a dictionary mapping an attribute
    to its features. Possible attributes are: INTENT, TEXT, ACTION_NAME,
    ACTION_TEXT, ENTITIES, SLOTS and ACTIVE_LOOP. Each attribute will be
    featurized into a list of `rasa.utils.features.Features`.
    """

    @abstractmethod
    def setup(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        pass

    @abstractmethod
    def featurize_state(self, state: State) -> Dict[Text, List["Features"]]:
        pass


class DefaultStateFeaturizer(SetupMixin):
    def __init__(self,) -> None:
        # TODO: parameter, pass on ...
        self.featurizer_using_interpreter = SubStateFeaturizerUsingInterpreter()
        self.fallback_featurizer = FallbackSubStateFeaturizer()
        self._setup_done = False

    def setup(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        """
        NOTE: domain and interpreter might not be known during instantiation

        Raises:
          RuntimeError if it has already been setup (cf. `is_ready`)
        """
        if self.configured:
            raise RuntimeError(
                f"Expected that {self.__class__.__name__} had not been configured before"
            )
        self.featurizer_using_interpreter.setup(domain=domain, interpreter=interpreter)
        self.fallback_featurizer.setup(domain)
        self._setup_done = True

    def is_setup(self) -> bool:
        return self._setup_done

    def _featurize_substate(
        self,
        sub_state: SubState,
        excluded_attributes: List[Text],
        attributes_requiring_sentence_feat: List[Text],
    ):
        """Featurization pipeline with special handling of given attribute.
        """
        substate_features = self.featurizer_using_interpreter.featurize(
            sub_state=sub_state, excluded_attributes=excluded_attributes,
        )
        for attribute in attributes_requiring_sentence_feat:
            attribute_features = substate_features.get(attribute, [])

            if not attribute_features or any(
                [f.type == Sequence for f in attribute_features]
            ):
                """
                TODO: existing sentence feature were (?) be overwritten, is this intended?
                """
                sentence_features = substate_featurizer.convert_sparse_sequence_to_sentence_features(
                    attribute_features
                )

            else:

                fallback_feats = self.fallback_featurizer.featurize(
                    sub_state, attribues=[attribute]
                )
                sentence_features = fallback_feats[attribute]

            substate_features[attribute] = [sentence_features]
        return substate_features

    def featurize_state(self, state: State) -> Dict[Text, List["Features"]]:
        """Encode the given state with the help of the given interpreter.

        Args:
            state: The state to encode

        Returns:
            A dictionary of state_type to list of features.
        """
        self.raise_if_ready_is(False)

        state_features = {}
        for substate_type, sub_state in state.items():

            # we distinguish three cases:
            if substate_type == USER and state_utils.previous_action_was_listen(state):

                substate_features = self._featurize_substate(
                    sub_state=sub_state,
                    excluded_attributes={ENTITIES},
                    attributes_requiring_sentence_feat=[INTENT],
                )

                if sub_state.get(ENTITIES):
                    substate_features[ENTITIES] = self.fallback_featurizer.featurize(
                        sub_state=sub_state, attributes=[ENTITIES],
                    )

            elif substate_type == PREVIOUS_ACTION:

                substate_features = self._featurize_substate(
                    sub_state=sub_state,
                    excluded_attributes={ENTITIES},
                    attributes_requiring_sentence_feat=[INTENT],
                )

            elif substate_type in {SLOTS, ACTIVE_LOOP}:
                substate_features = {
                    substate_type: self.self.fallback_featurizer.featurize(
                        sub_state=sub_state, attribute=substate_type,  # TODO:
                    )
                }
            else:
                substate_features = {}

            # TODO: this looks dangerous... attributes of substates must not overlap
            # (note: this was done like this before: https://github.com/RasaHQ/rasa/blob/main/rasa/core/featurizers/single_state_featurizer.py#L281)
            assert set(state_features.keys()).isdisjoint(substate_features.keys())

            state_features.update(substate_features)

        return state_features
