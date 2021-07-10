"""
NOTE:
- it would be great if we could turn the state featurizer into a combination of
  state item featurizers (cf. target featurizers)
-
"""

import logging
from rasa.shared.nlu.training_data.features import Features
import numpy as np
import scipy.sparse
from typing import List, Optional, Dict, Text, Any, Iterable

# from rasa.nlu.extractors.extractor import EntityTagSpec
# from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
# from rasa.nlu.utils import bilou_utils
from rasa.shared import io as io_utils
from rasa.shared.core.domain import SubState, State, Domain
from rasa.shared.core.domain import Domain
from rasa.shared.core import state as state_utils
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter
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


from rasa.core.featurizers.substate_featurizer import (
    FallbackFeaturizer,
    FeaturizerUsingInterpreter,
)


class StateFeaturizer:
    """Base class to transform the dialogue state into an ML format.

    Subclasses of SingleStateFeaturizer will decide how a bot will
    transform the dialogue state into a dictionary mapping an attribute
    to its features. Possible attributes are: INTENT, TEXT, ACTION_NAME,
    ACTION_TEXT, ENTITIES, SLOTS and ACTIVE_LOOP. Each attribute will be
    featurized into a list of `rasa.utils.features.Features`.
    """

    def __init__(self,) -> None:
        # TODO: parameter, pass on ...
        self.featurizer_using_interpreter = FeaturizerUsingInterpreter(...)
        self.fallback_featurizer = FallbackFeaturizer()

    def setup(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        """
        NOTE: domain and interpreter might not be known during instantiation

        Raises:
          RuntimeError if it has already been setup (cf. `is_ready`)
        """
        self.raise_if_ready_is(True)
        self.featurizer_using_interpreter.setup_domain(domain)
        self.fallback_featurizer.setup_domain(domain)
        self.featurizer_using_interpreter.setup_interpreter(interpreter)

    def is_ready(self) -> bool:
        return (
            self.featurizer_using_interpreter
            and self.featurizer_using_interpreter.is_ready()
        )

    def featurize_state(
        self, state: State, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:
        """Encode the given state with the help of the given interpreter.

        Args:
            state: The state to encode
            interpreter: The interpreter used to encode the state

        Returns:
            A dictionary of state_type to list of features.
        """
        self.raise_if_ready_is(False)
        interpreter = self.check_and_replace_interpreter_if_needed(interpreter)

        state_features = {}
        for substate_type, sub_state in state.items():

            # nlu pipeline didn't create features for user or action
            # this might happen, for example, when we have action_name in the state
            # but it did not get featurized because only character level
            # CountVectorsFeaturizer was included in the config.
            name_attribute = state_utils.get_name_attribute(sub_state)

            if substate_type == PREVIOUS_ACTION:
                substate_features = self.featurizer_using_interpreter.featurize(
                    # TODO
                    # sub_state,
                    # exclude={ENTITIES},
                    # keep_only_sparse_seq_converted_to_sent={INTENT, ACTION_NAME},
                    # use_fallback_interpreter={name_attribute},
                    # fallback_is_sparse=True,
                )

            elif substate_type == USER and state_utils.previous_action_was_listen(
                state
            ):
                # featurize user only if it is "real" user input,
                # i.e. input from a turn after action_listen
                substate_features = self.featurizer_using_interpreter.featurize(
                    # TODO
                    # sub_state,
                    # interpreter,
                    # exclude={ENTITIES},
                    # keep_only_sparse_seq_converted_to_sent={INTENT, ACTION_NAME},
                    # use_fallback_interpreter={name_attribute},
                    # fallback_is_sparse=True,
                )
                if sub_state.get(ENTITIES):
                    state_features[ENTITIES] = self.fallback_featurizer.featurize(
                        # sub_state=sub_state, attribute=ENTITIES, # TODO:
                    )

            elif substate_type in {SLOTS, ACTIVE_LOOP}:
                substate_features = {
                    substate_type: self.self.fallback_featurizer.featurize(
                        # sub_state=sub_state, attribute=substate_type, # TODO:
                    )
                }
            else:
                substate_features = {}

            state_features.update(substate_features)

        return state_features
