from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Text,
    Generic,
    Tuple,
    TypeVar,
    Set,
    Optional,
)
import logging

import numpy as np

from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.turns.to_dataset.label_from_turn_sequence_extractor import (
    LabelFromTurnsExtractor,
)
from rasa.core.turns.turn import Turn
from rasa.core.turns.to_dataset.turn_sub_sequence_generator import (
    TurnSubSequenceGenerator,
    steps2str,
)
from rasa.core.turns.to_dataset.featurizers import (
    TurnFeaturizer,
    LabelFeaturizer,
    LabelIndexer,
)
from rasa.core.turns.utils.trainable import Trainable
from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.features import Features

logger = logging.getLogger(__name__)

FeatureCollection = Dict[str, List[Features]]  # TODO: move this somewhere else
TurnType = TypeVar("TurnType")


class DatasetFromTurnSequenceGenerator(Generic[TurnType]):
    """Generates multiple turn-sequence+label pairs from a single sequence of turns."""

    def __init__(
        self,
        turn_sub_sequence_generator: TurnSubSequenceGenerator[TurnType],
        label_extractors: List[LabelFromTurnsExtractor[TurnType, Any]],
    ):
        self._turn_sub_sequence_generator = turn_sub_sequence_generator
        self._label_extractors = label_extractors or []
        self._ignore_duplicate_turn_label_pairs_cache: Optional[Set[int]] = None
        self._ignore_duplicate_turn_label_pairs_with: Optional[str] = None
        # sanity check
        self._label_names = list(
            extractor.name
            for extractor in self._label_extractors
            if extractor.name is not None
        )
        if len(self._label_names) > len(set(self._label_names)):
            raise ValueError(
                "Expected the names attached to the label "
                "extractors to have unique names."
            )

    def ignore_duplicate_turn_label_pairs_during_training(
        self, label_name: Optional[str] = None, cache: Optional[Set[int]] = None
    ):
        self._ignore_duplicate_turn_label_pairs_cache = (
            cache if (cache is not None) else set()
        )
        self._ignore_duplicate_turn_label_pairs_with = label_name

    @property
    def label_names(self) -> List[str]:
        return self._label_names

    def train_indexer_for(
        self, indexer: LabelIndexer, domain: Domain, extractor_idx: int
    ) -> None:
        # a workaround to not make _label_extractors public
        indexer.train(domain=domain, extractor=self._label_extractors[extractor_idx])

    def apply_to(
        self,
        turns: List[TurnType],
        training: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Tuple[List[Turn], Optional[Dict[Text, Any]]]]:

        for processed_turns in self._turn_sub_sequence_generator.apply_to(
            turns=turns,
            training=training,
            context=context,
        ):
            logger.debug(
                f"Start extracting next turn sequence from:\n{steps2str(processed_turns)}"
            )
            processed_turns, labels = LabelFromTurnsExtractor.apply_all(
                label_extractors=self._label_extractors,
                turns=processed_turns,
                training=training,
                inplace_allowed=(not training),
            )

            if self._ignore_duplicate_turn_label_pairs_cache is not None:
                if self._ignore_duplicate_turn_label_pairs_with is not None:
                    labels_for_hashing = labels.get(
                        self._ignore_duplicate_turn_label_pairs_with, None
                    )
                else:
                    labels_for_hashing = tuple(
                        labels.get(label_name, None) for label_name in self._label_names
                    )
                identifier = hash(
                    (
                        tuple(
                            DialogueStateTracker.freeze_current_state(turn.state)
                            for turn in processed_turns
                        ),
                        labels_for_hashing,
                    )
                )

                if identifier in self._ignore_duplicate_turn_label_pairs_cache:
                    logger.debug(
                        "Continue (duplicate of other turn sequence and label pair)"
                    )
                    continue
                else:
                    self._ignore_duplicate_turn_label_pairs_cache.add(identifier)

            logger.debug(
                f"Done extracting next turn Sequence: \n"
                f"{steps2str(processed_turns)}\nWith raw labels: \n{labels}"
            )
            yield processed_turns, labels


class DatasetFeaturizer(Trainable, Generic[TurnType]):
    """Generates several featurized turn-sequence+label pairs from one turn sequence."""

    def __init__(
        self,
        turn_featurizer: TurnFeaturizer[TurnType],
        label_encoders: Dict[
            str, Tuple[Optional[LabelFeaturizer], Optional[LabelIndexer]]
        ],
    ):
        super().__init__()
        self.mark_as_not_trained()
        self._label_encoders = label_encoders
        self._turn_featurizer = turn_featurizer

    def train(
        self,
        domain: Domain,
        dataset_generator: DatasetFromTurnSequenceGenerator,
    ) -> None:
        self._assert_label_names_match_with(dataset_generator)
        self._turn_featurizer.train(domain=domain)
        for idx, label_name in enumerate(dataset_generator.label_names):
            (featurizer, indexer) = self._label_encoders[label_name]
            if featurizer:
                featurizer.train(domain=domain)
            if indexer:
                dataset_generator.train_indexer_for(
                    indexer=indexer, domain=domain, extractor_idx=idx
                )

        self.mark_as_trained()

    def _assert_label_names_match_with(
        self, dataset_generator: DatasetFromTurnSequenceGenerator
    ) -> None:
        label_names = set(dataset_generator.label_names)
        missing_extractor = set(self._label_encoders.keys()).difference(label_names)
        if missing_extractor:
            raise ValueError(
                f"Expected all label encoders to have some corresponding "
                f"label extractor. Extractor is missing for "
                f"{missing_extractor}."
            )
        missing_encoder = label_names.difference(self._label_encoders.keys())
        if missing_encoder:
            raise ValueError(
                f"Expected all extracted label to be encoded in some way. "
                f"Encoders are "
                f"missing for {missing_encoder}."
            )

    def apply_to(
        self,
        input_turns: List[Turn],
        labels: Optional[Dict[Text, Any]],
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Tuple[List[FeatureCollection], FeatureCollection, Dict[Text, np.ndarray]]:
        self.raise_if_not_trained()

        collected_features = {}
        collected_indices = {}

        if labels is not None:
            for label_name, raw_label in labels.items():
                featurizer, indexer = self._label_encoders[label_name]
                if featurizer:
                    collected_features[label_name] = featurizer.featurize(
                        raw_label=raw_label,
                        precomputations=precomputations,
                    )
                if indexer:
                    collected_indices[label_name] = indexer.index(raw_label=raw_label)

            logger.debug(f"Featurized raw labels : {collected_features}")
            logger.debug(f"Indexed raw labels: {collected_indices}")

        # featurize the (remaining) input (during training)
        input_featurized = [
            self._turn_featurizer.featurize(turn, precomputations=precomputations)
            for turn in input_turns
        ]
        logger.debug(f"Featurized state:\n" f"{steps2str(input_featurized)}")
        return input_featurized, collected_features, collected_indices
