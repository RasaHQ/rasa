from __future__ import annotations

from typing import (
    List,
    Optional,
    Dict,
    Text,
    Union,
    Tuple,
)

import scipy.sparse

from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.turns.state.state import ExtendedState
from rasa.core.turns.utils.feature_lookup import FeatureLookup
from rasa.core.turns.utils.multi_label_encoder import MultiLabelEncoder
from rasa.core.turns.to_dataset.featurizers import TurnFeaturizer
from rasa.shared.core.constants import ACTIVE_LOOP, PREVIOUS_ACTION, SLOTS, USER
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import (
    ACTION_NAME,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    INTENT,
)
from rasa.shared.nlu.training_data.features import Features


class BasicStateFeaturizer(TurnFeaturizer[ExtendedState]):
    """Featurize a single state.

    We assume here that the events that have been used to generate the Stateful
    turns have been augmented with NLU information already (i.e. via annotations
    in the training data or via the application of an NLU pipeline during prediction
    time).
    That is, we solely used to turn information into `Features` here.

    More specifically, we require that:
    1. a turn's state always contains an action name in its 'previous action'
       substate - if there is an previous action substate - and an intent in
       its 'user' substate - if there is a user substate.
    2. the attributes of all substates are mutually different.
    """

    def _train(
        self,
        domain: Domain,
    ) -> None:
        self._encoders = {
            attribute: MultiLabelEncoder(dimension_names=dimension_names)
            for attribute, dimension_names in [
                (INTENT, domain.intents),
                # NOTE: even though this contains action names and texts, we will
                # only create multihot encodings for action names
                (ACTION_NAME, domain.action_names_or_texts),
                (ENTITIES, domain.entity_states),
                (SLOTS, domain.slot_states),
                (ACTIVE_LOOP, domain.form_names),
            ]
        }

    def _featurize(
        self,
        turn: ExtendedState,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        training: bool = True,
    ) -> Dict[Text, List[Features]]:
        """Encodes the given state with the help of the given precomputed features.

        We assume here that the attributes of all substates are mutually different.

        Args:
            turn: The state to encode
            precomputations:

        Returns:
            A dictionary of state_type to list of features.
        """
        self.raise_if_not_trained()

        state = turn.state
        features = {}

        # We always featurize the previous action substate. However, if the turn is
        # attributed to the user, then the previous action will just be an
        # `ActionListen`.
        if PREVIOUS_ACTION in state:
            sub_state = state[PREVIOUS_ACTION]

            # Attempt to featurize the substate via the interpreter, but...
            features_via_interpreter = FeatureLookup.lookup_features(
                sub_state, precomputations=precomputations
            )
            features.update(features_via_interpreter)

            # ... note that the only featurizer that could featurize this attribute
            # is the `CountVectorizer`, which will create sparse sequence and sentence
            # features.
            # We replace these features with some sparse sentence features:
            if ACTION_NAME in features:
                features[ACTION_NAME] = self._sum_sparse_sequence_features(
                    features[ACTION_NAME]
                )

            # If there is an action name but no features yet, we create some:
            if ACTION_NAME not in features:
                action_name = sub_state.get(ACTION_NAME, None)
                if action_name:
                    features[ACTION_NAME] = [
                        self._multihot_encoding_for_action_name(action_name=action_name)
                    ]

        # Remember that the user state might be empty despite the turn belonging to
        # the user because of the edge case where a turn just captures an action listen.
        if turn.get_type() == ExtendedState.USER and USER in state:

            sub_state = state[USER]

            # Only featurize attributes != entities via the input pipeline
            not_featurized_via_interpreter = {ENTITIES}

            # Create some features via the interpreter
            features_via_interpreter = FeatureLookup.lookup_features(
                sub_state,
                precomputations=precomputations,
                exclude_from_results=not_featurized_via_interpreter,
            )
            features.update(features_via_interpreter)

            # Use a simple multihot encoding for entities
            if ENTITIES in sub_state:
                features[ENTITIES] = [
                    self._multihot_encoding_for_entities(sub_state.get(ENTITIES))
                ]

            # The only featurizer that could featurize this attribute is the
            # CountVectorizer, which will create sparse sequence and sentence
            # features.
            # We replace these features with some sparse sentence features:
            if INTENT in features:
                features[INTENT] = self._sum_sparse_sequence_features(features[INTENT])

            # If there is an intent but no features yet, we create some:
            if INTENT not in features:
                intent = sub_state.get(INTENT, None)
                if intent:
                    features[INTENT] = [self._multihot_encoding_for_intent(intent)]

        # We always featurize the context, i.e. slots and loop information.
        if SLOTS in state:
            features[SLOTS] = [
                self._multihot_encoding_for_slots_as_features(
                    slots_as_features=state[SLOTS]
                )
            ]
        if ACTIVE_LOOP in state:
            features[ACTIVE_LOOP] = [
                self._multihot_encoding_for_active_loop(
                    active_loop_name=state[ACTIVE_LOOP]["name"]
                )
            ]
        return features

    def _multihot_encoding_for_intent(self, intent: Text) -> Features:
        return self._encoders[INTENT].encode_as_sparse_sentence_feature(
            {intent: 1}, attribute=INTENT, origin=self.__class__.__name__
        )

    def _multihot_encoding_for_action_name(
        self, action_name: Optional[Text]
    ) -> Features:
        return self._encoders[ACTION_NAME].encode_as_sparse_sentence_feature(
            {action_name: 1} if action_name else {},
            attribute=ACTION_NAME,
            origin=self.__class__.__name__,
        )

    def _multihot_encoding_for_entities(self, entities: List[Text]) -> Features:
        return self._encoders[ENTITIES].encode_as_sparse_sentence_feature(
            {entity: 1 for entity in (entities or [])},
            attribute=ENTITIES,
            origin=self.__class__.__name__,
        )

    def _multihot_encoding_for_active_loop(self, active_loop_name: Text) -> Features:
        return self._encoders[ACTIVE_LOOP].encode_as_sparse_sentence_feature(
            {active_loop_name: 1}, attribute=ACTIVE_LOOP, origin=self.__class__.__name__
        )

    def _multihot_encoding_for_slots_as_features(
        self, slots_as_features: Dict[Text, Union[Text, Tuple[float, ...]]]
    ) -> Features:
        # cf. `slot_states` in `domain.py`
        dim_to_value = {
            f"{slot_name}_{i}": value
            for slot_name, slot_as_feature in slots_as_features.items()
            for i, value in enumerate(slot_as_feature)
            if isinstance(slot_as_feature, list)
        }
        return self._encoders[SLOTS](
            dim_to_value, attribute=SLOTS, origin=self.__class__.__name__
        )

    @staticmethod
    def _sum_sparse_sequence_features(
        features: List[Features],
    ) -> List[Features]:
        return [
            Features(
                scipy.sparse.coo_matrix(feature.features.sum(0)),
                FEATURE_TYPE_SENTENCE,
                feature.attribute,
                feature.origin,
            )
            for feature in features
            if feature.is_sparse() and feature.feature_type == FEATURE_TYPE_SEQUENCE
        ]
