from typing import Tuple, List, Optional, Dict, Text, Any, Set, Iterator
import numpy as np
from dataclasses import dataclass
import logging

from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.shared.core.domain import State
from rasa.core.turns.to_dataset.dataset_from_turn_sequence import (
    TurnFeaturizer,
    DatasetFeaturizer,
)
from rasa.core.turns.utils.trainable import Trainable
from rasa.core.turns.state.state import ExtendedState, ExtendedStateParser
from rasa.core.turns.state.label_from_state_sequence_extractor import (
    ExtractActionFromLastState,
    ExtractEntitiesFromLastUserState,
)
from rasa.core.turns.state.state_sequence_modifiers import (
    IfLastStateWasUserStateKeepEitherTextOrNonText,
    RemoveStatesWithPrevActionUnlikelyIntent,
    RemoveUserTextIfIntentFromEveryState,
)
from rasa.core.turns.to_dataset.featurizers import (
    FeaturizeViaEntityTagsEncoder,
    IndexerFromLabelExtractor,
)
from rasa.core.turns.to_dataset.turn_sub_sequence_generator import (
    KeepMaxHistory,
    HasMinLength,
)

from rasa.core.turns.to_dataset.dataset_from_turn_sequence import (
    DatasetFromTurnSequenceGenerator,
)
from rasa.core.turns.to_dataset.turn_sub_sequence_generator import (
    TurnSubSequenceGenerator,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import ACTION_NAME, ENTITY_TAGS
from rasa.shared.nlu.training_data.features import Features

logger = logging.getLogger(__name__)

USE_TEXT_FOR_LAST_USER_INPUT = "use_text_for_last_user_input"


@dataclass
class InputStatesCreatorForNextActionPrediction:

    max_history: Optional[int] = None
    ignore_duplicate_turn_label_pairs: bool = True
    ignore_action_unlikely_intent: bool = False
    ignore_rule_only_turns: bool = False
    rule_only_data: Optional[Dict[Text, Any]] = None

    def __post_init__(self):

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

        filters = [HasMinLength(2)]

        modifiers = [
            KeepMaxHistory(max_history=self.max_history, offset_for_training=+1),
            IfLastStateWasUserStateKeepEitherTextOrNonText(
                keep_text=lambda turns, context, training: training
                or context[USE_TEXT_FOR_LAST_USER_INPUT]
            ),
            RemoveUserTextIfIntentFromEveryState(on_training=False),
        ]

        turn_sequence_generator = TurnSubSequenceGenerator(
            preprocessing=preprocessing,
            filters=filters,
            ignore_duplicates=False,
            modifiers=modifiers,
            result_filters=None,
        )

        # TODO: There are no *unit tests* to test label extraction during training time
        #  but we can test this via RulePolicy.
        extractor_pipeline = [
            # ExtractIntentFromLastUserState(name=INTENT),
            ExtractEntitiesFromLastUserState(name=ENTITY_TAGS),
            ExtractActionFromLastState(name=ACTION_NAME, remove_last_turn=True),
        ]

        self._dataset_generator = DatasetFromTurnSequenceGenerator(
            turn_sub_sequence_generator=turn_sequence_generator,
            label_extractors=extractor_pipeline,
        )

    def create_training_data(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[Text]]:
        """

        used by rule policies during training (no unit tests for this)

        """
        # TODO: There are no *unit tests* to test label extraction during training time.
        #  but we could test this via RulePolicy.
        ...

    def stream_training_turn_sequences(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Iterator[Tuple[List[State], Text]]:

        if self.ignore_duplicate_turn_label_pairs:
            cache: Set[int] = set()
            self._dataset_generator.ignore_duplicate_turn_label_pairs_during_training(
                cache=cache,
                label_name=ACTION_NAME,
            )

        for tracker in trackers:

            # Note: During training, we assume that no rule only data / rule turns
            # need to be removed.
            dialogue_states = self._training_state_parser.parse(
                tracker=tracker,
                domain=domain,
            )

            yield from self._dataset_generator.apply_to(
                turns=dialogue_states,
                training=True,
                context={USE_TEXT_FOR_LAST_USER_INPUT: True},
            )

        # Reminder: This cache needs clearing because it's attached to the generator.
        if self.ignore_duplicate_turn_label_pairs:
            cache.clear()

    def create_inference_turn_sequence(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[State]:
        """

        used by rule-policies during inference

        """
        turns = self._inference_state_parser.parse(tracker, domain)
        modified_turns, _ = next(
            self._dataset_generator.apply_to(
                turns,
                training=False,
                context={USE_TEXT_FOR_LAST_USER_INPUT: use_text_for_last_user_input},
            )
        )
        return modified_turns

    def create_inference_data(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[State]:
        """

        used by rule-policies during inference

        """
        modified_turns = self.create_inference_turn_sequence(tracker, domain)
        return [turn.state for turn in modified_turns]


@dataclass
class InputFeaturesCreatorForNextActionPrediction(Trainable):

    state_featurizer: TurnFeaturizer[ExtendedState]
    max_history: Optional[int] = None
    ignore_duplicate_turn_label_pairs: bool = True
    bilou_tagging: bool = False
    ignore_action_unlikely_intent: bool = False
    ignore_rule_only_turns: bool = False
    rule_only_data: Optional[Dict[Text, Any]] = None

    def __post_init__(self) -> None:
        self.mark_as_not_trained()

        self._input_states = InputStatesCreatorForNextActionPrediction(
            max_history=self.max_history,
            ignore_duplicate_turn_label_pairs=self.ignore_duplicate_turn_label_pairs,
            ignore_action_unlikely_intent=self.ignore_action_unlikely_intent,
        )
        # TODO: fix this, needed for training the featurizers atm
        self._dataset_generator = self._input_states._dataset_generator

        label_encoders = {
            ACTION_NAME: (
                None,
                IndexerFromLabelExtractor(),
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
        """

        Args:
          trackers:
          domain:
          precomputations:
        """
        self.raise_if_not_trained()
        inputs = []
        labels_action = []
        labels_entities = []

        for (
            input_states,
            labels,
        ) in self._input_states.stream_training_turn_sequences(
            trackers=trackers,
            domain=domain,
        ):
            featurized = self._dataset_featurizer.apply_to(
                input_turns=input_states,
                labels=labels,
                precomputations=precomputations,
            )
            inputs.append(featurized[0])
            labels_action.append(featurized[2][ACTION_NAME])  # only indices
            labels_entities.append(featurized[1][ENTITY_TAGS])  # only Features

        # TODO: we could just yield these one by one / make this an iterable or even
        #  map-style torch-like data set (but it's out of scope for now / currently
        #  everything expects these in memory)
        return inputs, np.array(labels_action), labels_entities

    def create_inference_data(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        use_text_for_last_user_input: bool = False,
    ) -> Dict[Text, List[Features]]:
        self.raise_if_not_trained()

        input_states = self._input_states.create_inference_turn_sequence(
            tracker=tracker,
            domain=domain,
            use_text_for_last_user_input=use_text_for_last_user_input,
        )
        (input_features, _, _,) = self._dataset_featurizer.apply_to(
            input_turns=input_states, labels=None, precomputations=precomputations
        )

        return input_features
