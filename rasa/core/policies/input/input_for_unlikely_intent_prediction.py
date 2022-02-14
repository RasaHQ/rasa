from typing import Tuple, List, Optional, Dict, Text, Any, Set
import numpy as np
from dataclasses import dataclass
import logging

from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.turns.to_dataset.label_from_turn_sequence_extractor import (
    FakeLabelFromTurnsExtractor,
)
from rasa.shared.core.domain import State

from rasa.core.turns.utils.trainable import Trainable
from rasa.core.turns.to_dataset.turn_sub_sequence_generator import (
    TurnSubSequenceGenerator,
    steps2str,
    KeepMaxHistory,
    EndsWith,
    HasMinLength,
    RemoveLastTurn,
)
from rasa.core.turns.to_dataset.dataset_from_turn_sequence import (
    DatasetFromTurnSequenceGenerator,
    DatasetFeaturizer,
)
from rasa.core.turns.to_dataset.featurizers import TurnFeaturizer
from rasa.core.turns.state.state import ExtendedState, ExtendedStateParser
from rasa.core.turns.state.label_from_state_sequence_extractor import (
    ExtractIntentFromLastUserState,
    ExtractEntitiesFromLastUserState,
)
from rasa.core.turns.state.state_sequence_modifiers import (
    IfLastStateWasUserStateKeepEitherTextOrNonText,
    RemoveStatesWithPrevActionUnlikelyIntent,
    RemoveUserTextIfIntentFromEveryState,
)
from rasa.core.turns.to_dataset.featurizers import (
    MultiIndexerFromLabelExtractor,
    FeaturizeViaEntityTagsEncoder,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import ENTITY_TAGS, INTENT
from rasa.shared.nlu.training_data.features import Features

logger = logging.getLogger(__name__)

USE_TEXT_FOR_LAST_USER_INPUT = "use_text_for_last_user_input"
FEATURIZER_FILE = "featurizer.file"  # TODO


@dataclass
class InputFeaturesCreatorForUnlikelyIntentActionPrediction(Trainable):
    """

    Note: Current solution doesn't allow for streaming right away. One pass to
       compute the labels is needed because for some reason, positive labels are
       collected to create multi-label encodings.
    """

    state_featurizer: TurnFeaturizer[ExtendedState]
    max_history: Optional[int] = None
    ignore_duplicate_input_turns: bool = True
    bilou_tagging: bool = False
    ignore_action_unlikely_intent: bool = False
    ignore_rule_only_turns: bool = False
    rule_only_data: Optional[Dict[Text, Any]] = None

    def __post_init__(self) -> None:
        self.mark_as_not_trained()

        # Note: During inference, events that are the result of a rule data
        # application will be tagged accordingly. We ignore such turns here,
        # under the assumption that we did not see such turns during training.
        # (Which is not really true).
        self._inference_state_parser = ExtendedStateParser(
            omit_unset_slots=False,
            ignore_rule_only_turns=self.ignore_rule_only_turns,
            rule_only_data=self.rule_only_data,
        )
        self._training_state_parser = ExtendedStateParser(
            omit_unset_slots=False,
            ignore_rule_only_turns=False,  # not done during training
            rule_only_data=False,  # not done during training
        )

        preprocessing = (
            [RemoveStatesWithPrevActionUnlikelyIntent()]
            if self.ignore_action_unlikely_intent
            else []
        )

        filters = [
            HasMinLength(2),
            EndsWith(turn_type=ExtendedState.USER),
        ]

        # For historic reasons, our interpretation of the "max history" needs
        # to be shifted by +1 ...
        max_history = self.max_history + 1 if self.max_history is not None else None
        modifiers = [
            KeepMaxHistory(max_history=max_history, offset_for_training=0),
            IfLastStateWasUserStateKeepEitherTextOrNonText(
                keep_text=lambda turns, context, training: training
                or context[USE_TEXT_FOR_LAST_USER_INPUT]
            ),
            RemoveUserTextIfIntentFromEveryState(on_training=False),
        ]

        turn_sequence_generator = TurnSubSequenceGenerator(
            preprocessing=preprocessing,
            filters=filters,
            ignore_duplicates=False,  # see: create_train_data
            modifiers=modifiers,
            result_filters=None,
        )

        extractor_pipeline = [
            ExtractIntentFromLastUserState(name=INTENT),
            ExtractEntitiesFromLastUserState(name=ENTITY_TAGS),
            FakeLabelFromTurnsExtractor(
                RemoveLastTurn(on_training=True, on_inference=True)
            ),
        ]

        self._dataset_generator = DatasetFromTurnSequenceGenerator(
            turn_sub_sequence_generator=turn_sequence_generator,
            label_extractors=extractor_pipeline,
        )

        label_encoders = {
            INTENT: (
                None,
                MultiIndexerFromLabelExtractor(),
            ),  # no featurizer
            ENTITY_TAGS: (
                FeaturizeViaEntityTagsEncoder(bilou_tagging=self.bilou_tagging),
                None,  # no indexer
            ),
        }

        self._dataset_featurizer = DatasetFeaturizer(
            label_encoders=label_encoders,
            turn_featurizer=self.state_featurizer,
        )

    def train(self, domain: Domain) -> None:
        self._dataset_featurizer.train(
            domain=domain,
            dataset_generator=self._dataset_generator,
        )
        self.mark_as_trained()

    def create_training_data(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
        List[List[Dict[Text, List[Features]]]],
    ]:
        self.raise_if_not_trained()

        inputs_features = []
        inputs_state_hash = []
        labels_entities_features = []

        state_hash_to_multi_label_map: Dict[int, Set[str]] = {}
        state_hash_set: Set[int] = set()

        for tracker in trackers:

            dialogue_states = self._training_state_parser.parse(
                tracker=tracker,
                domain=domain,
            )

            logger.debug(f"states: {dialogue_states}")

            for (input_states, labels,) in self._dataset_generator.apply_to(
                turns=dialogue_states,
                training=True,
                context={USE_TEXT_FOR_LAST_USER_INPUT: True},
            ):

                # We use the hash of the input state sequence to determine all possible
                # intent labels - before also using it to de-duplicate the data.
                input_states_hash = hash(
                    tuple(
                        DialogueStateTracker.freeze_current_state(turn.state)
                        for turn in input_states
                    )
                )

                # We *always* collect all possible labels that an input state sequence
                # can have and combine them later into a multi-label annotation that is
                # assigned to either
                # - *all* instances of that input state sequence if we don't ignore
                #   duplicates, or
                # - the only instance of that input state sequence that is left after
                #   we have de-duplicated the data
                multi_label = state_hash_to_multi_label_map.setdefault(
                    input_states_hash, set()
                )
                multi_label.add(labels[INTENT])

                if self.ignore_duplicate_input_turns:
                    if input_states_hash in state_hash_set:
                        continue
                state_hash_set.add(input_states_hash)

                featurized = self._dataset_featurizer.apply_to(
                    input_turns=input_states,
                    labels={ENTITY_TAGS: labels[ENTITY_TAGS]},
                    precomputations=precomputations,
                )
                inputs_features.append(featurized[0])
                inputs_state_hash.append(input_states_hash)  # map to multi-intent later
                labels_entities_features.append(featurized[1][ENTITY_TAGS])

                logger.debug(">" * 100)
                logger.debug(f"inputs: {steps2str(featurized[0])}")
                logger.debug(f"labels: {labels[INTENT]}")
                logger.debug(f"entities: {featurized[1][ENTITY_TAGS]}")
                logger.debug("<" * 100)

        # Assign the multi-labels to all instances of the state sequence that are
        # left (see above), featurize, and pad them.
        # TODO: why is the padding done here?
        labels_intents_multi_label = [
            self._dataset_featurizer.apply_to(
                input_turns=[],
                labels={INTENT: list(state_hash_to_multi_label_map[state_hash])},
                precomputations=precomputations,
            )[2][INTENT]
            for state_hash in inputs_state_hash
        ]
        max_number_of_labels = max(
            len(multi_label) for multi_label in labels_intents_multi_label
        )
        # FIXME: common padding function somewhere
        labels_intents_multi_label_padded = [
            np.concatenate(
                (multi_label, [-1] * (max_number_of_labels - len(multi_label)))
            )
            for multi_label in labels_intents_multi_label
        ]
        labels_array = np.array(labels_intents_multi_label_padded)

        return inputs_features, labels_array, labels_entities_features

    def create_inference_data(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        use_text_for_last_user_input: bool = False,
    ) -> Dict[Text, List[Features]]:
        self.raise_if_not_trained()

        states = self._inference_state_parser.parse(tracker, domain)
        logger.debug(f"states: {states}")

        input_states, _ = next(
            self._dataset_generator.apply_to(
                turns=states,
                training=False,
                context={USE_TEXT_FOR_LAST_USER_INPUT: use_text_for_last_user_input},
            )
        )
        input_features, _, _ = self._dataset_featurizer.apply_to(
            input_turns=input_states, labels=None, precomputations=precomputations
        )

        return input_features

    def _create_inference_states_only(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[State]:
        """This is not used in any policy. We just keep it alive for tests."""
        turns = self._inference_state_parser.parse(tracker, domain)
        modified_turns, _ = next(
            self._dataset_generator.apply_to(
                turns,
                training=False,
                context={USE_TEXT_FOR_LAST_USER_INPUT: use_text_for_last_user_input},
            )
        )
        return [turn.state for turn in modified_turns]
