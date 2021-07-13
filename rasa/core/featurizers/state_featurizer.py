import logging
from typing import (
    List,
    Optional,
    Dict,
    Text,
)
from typing_extensions import TypedDict

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
    FEATURE_TYPE_SEQUENCE,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.core.featurizers import substate_featurizer
from rasa.core.featurizers.substate_featurizer import (
    SubstateFeaturizerUsingMultiHotVectors,
    SetupMixin,
    SubStateFeaturizerUsingInterpreter,
)

logger = logging.getLogger(__name__)


USE_INTERPRETER = "use_interpreter"
USE_MULTIHOT = "use_multihot"
ENFORCE_SENTENCE_FEATURE = "enforce_sentence_feature"
ALL_METHODS = [USE_INTERPRETER, USE_MULTIHOT, ENFORCE_SENTENCE_FEATURE]


SubtypeToAttributeToMethod = TypedDict(
    "SubtypeToAttributeToMethod",
    {
        USER: Dict[Text, List[Text]],
        PREVIOUS_ACTION: Dict[Text, List[Text]],
        ACTIVE_LOOP: bool,
        SLOTS: bool,
    },
)


def convert_to_subtype_to_method_to_attribute_map(
    config: SubtypeToAttributeToMethod,
) -> Dict[Text, Dict[Text, List[Text]]]:
    """Convert config to a mapping from state type and method to attribute list ."""
    return {
        {
            method: [
                attribute
                for attribute, usage_config in config[sub_type].items()
                if method in usage_config
            ]
            for method in ALL_METHODS
        }
        for sub_type in SubtypeToAttributeToMethod.__annotations__
    }


class StateFeaturizer(SetupMixin):
    """

    """

    def __init__(
        self,
        config: SubtypeToAttributeToMethod,
        featurizer_using_interpreter: Optional[
            SubStateFeaturizerUsingInterpreter
        ] = None,
        featurizer_using_multihot: Optional[
            SubstateFeaturizerUsingMultiHotVectors
        ] = None,
    ) -> None:
        self._config = config
        self._config_rearranged = convert_to_subtype_to_method_to_attribute_map(config)
        self._featurizer_using_interpreter = featurizer_using_interpreter
        self._featurizer_using_multihot = featurizer_using_multihot

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
        self._featurizer_using_interpreter = SubStateFeaturizerUsingInterpreter()
        self._featurizer_using_interpreter.setup(domain=domain, interpreter=interpreter)
        self._featurizer_using_multihot = SubstateFeaturizerUsingMultiHotVectors()
        self._featurizer_using_multihot.setup(domain)

    def is_setup(self) -> bool:
        return (
            self._featurizer_using_interpreter.is_setup()
            and self._featurizer_using_multihot.is_setup()
        )

    def featurize_state(self, state: State) -> Dict[Text, List["Features"]]:
        """Encode the given state with the help of the given interpreter.

        Args:
            state: The state to encode

        Returns:
            A dictionary of state_type to list of features.
        """
        self.raise_if_ready_is(False)

        attribute_feature_map = {}
        for substate_type, sub_state in state.items():

            # skip this # TODO: when does this happen?
            if substate_type == USER and not state_utils.previous_action_was_listen(
                state
            ):
                continue

            methods_to_attributes = self._config_rearranged.get(substate_type, [])

            substate_features = {}

            for method in [USE_INTERPRETER, USE_MULTIHOT]:

                new_features = self._featurizer_using_interpreter.featurize(
                    sub_state=sub_state, attributes=methods_to_attributes[method],
                )
                for attribute, feat in new_features:
                    feature_list = substate_features.setdefault(attribute, [])
                    feature_list.append(feat)

            if ENFORCE_SENTENCE_FEATURE in methods_to_attributes:

                self._autofill_sentence_feature(
                    substate_features=substate_features,
                    attribute=methods_to_attributes[ENFORCE_SENTENCE_FEATURE],
                )

            attribute_feature_map.update(substate_features)

        # NOTE: checking whether all expected attributes are populated here makes
        # no sense since that would require the interpreter to add some indicator
        # for "tried but didn't find any (e.g. entities)" here

        return attribute_feature_map

    def _autofill_sentence_feature(
        self,
        sub_state: SubState,
        substate_features: Dict[Text, Features],
        attribute: Text,
    ):
        """

        """
        attribute_features = substate_features.get(attribute, [])
        if any([f.type == FEATURE_TYPE_SEQUENCE for f in attribute_features]):
            # yeay, we have sentence features already
            # FIXME: this is different from current pipeline
            pass

        elif any([f.type == FEATURE_TYPE_SEQUENCE for f in attribute_features]):
            # we can deduce them from sequence features...
            sentence_features = substate_featurizer.convert_sparse_sequence_to_sentence_features(
                attribute_features
            )
        else:
            # we have nothing at all...
            fallback_feats = self._featurizer_using_multihot.featurize(
                sub_state, attribues=[attribute]
            )
            sentence_features = fallback_feats[attribute]

        substate_features[attribute] = [sentence_features]
        return substate_features
