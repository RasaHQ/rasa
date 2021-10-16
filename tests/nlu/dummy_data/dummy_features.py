import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Set, Text, List, Optional

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import numpy as np
import scipy.sparse
from rasa.nlu.constants import TOKENS_NAMES

# TODO: from rasa.nlu.constants import SENTENCE_FEATURES, SEQUENCE_FEATURES  (?)

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.model_data import FeatureArray


@dataclass
class FeaturizerDescription:
    """A description of a featurizer.

    Args:
        name: featurizer name
        sequence_dim: feature dimension of sequence features
        sentence_dim: feature dimension of sentence features
        is_sparse: determines whether it produces sparse features
        sequence_attributes: the attributes for which it produced sequence features
        sentence_attributes: the attributes for which it produces sentence features
    """

    name: Text = "featurizer"
    sequence_dim: int = 4
    sentence_dim: int = 4
    is_sparse: bool = True
    sequence_attributes: Set[Text] = field(default_factory=lambda: set())
    sentence_attributes: Set[Text] = field(default_factory=lambda: set())

    @property
    def attributes(self):
        return self.sequence_attributes.union(self.sentence_attributes)


# Each featurizer can produce sequence and/or sentence features.
Featurization = TypedDict(
    "Featurization",
    {FEATURE_TYPE_SENTENCE: Features, FEATURE_TYPE_SEQUENCE: Features},
    total=False,
)

# Our neural nets expect that the features produced by various featurizers have
# been concatenated already. For each message, we concatenate the featurizations
# for each combination of feature type and sparseness property:
ConcatenatedFeaturizations = TypedDict(
    "ConcatenatedFeaturizations",
    {
        (FEATURE_TYPE_SENTENCE, True): scipy.sparse.spmatrix,
        (FEATURE_TYPE_SEQUENCE, True): scipy.sparse.spmatrix,
        (FEATURE_TYPE_SENTENCE, False): np.ndarray,
        (FEATURE_TYPE_SEQUENCE, False): np.ndarray,
    },
    total=False,
)


@dataclass
class DummyFeatures:
    """Generates pseudo-random features.

    It can generate a pseudo-random feature values that are determined
    by the attribute of a message, a featurizer, the feature type and level.
    Assuming the given message texts are unique, this enables us to identify,
    with high probability, which features after they have been concatenated or
    stacked in various ways.

    This feature generator allows for imitating a featurization where ...
    1. each featurizer may produce sparse and/or dense features
    2. if a featurizer produces features of a certain kind (sparse/dense), then it
        produces a single sequence and/or a single sentence features of that kind
    3. each featurizer can produces features for an attribute if and only if that
       attribute has been tokenized (i.e. the list of tokens is not empty) and will
       produce sequence features with a sequence length that is equal to the number
       of tokens
    4. featurizers are always applied exactly once to a message - in the exact same
       sequential order

    Args:
        featurizer_descriptions: list of featurizer descriptions
        used_featurizers: list of featurizers which will be considered when
            constructing the expected concatenated features
    """

    featurizer_descriptions: List[FeaturizerDescription] = field(
        default_factory=lambda: [FeaturizerDescription()]
    )
    SEED_RANGE = 10 ** 5

    @staticmethod
    def generate_pseudo_random_value(seed: List[Optional[Text]]) -> int:
        """Creates a pseudo-random value using the given parameters as seed."""
        rng = random.Random("".join([f"{item}" for item in seed]))
        return rng.randint(0, DummyFeatures.SEED_RANGE)

    @staticmethod
    def generate_pseudo_random_feature(
        attribute: Text,
        tokens: List[Text],
        origin: Text,
        is_sparse: bool,
        feature_type: Text,
        feature_dim: int,
    ) -> Features:
        """Imitates the creation of a single `Features` by a featurizer."""
        # determine the shape
        seq_len = 1 if feature_type == FEATURE_TYPE_SENTENCE else len(tokens)
        shape = (seq_len, feature_dim)
        # create a unique seed value and initialise an rng
        seed = DummyFeatures.generate_pseudo_random_value(
            [attribute, tokens, origin, feature_type,]
        )
        rng = np.random.default_rng(seed)
        # fill a sparse/dense matrix using the rng
        if not is_sparse:
            matrix = rng.random(shape)
        else:
            matrix = scipy.sparse.eye(m=shape[0], n=shape[1]) * rng.random()
        # add it to the message if required
        feature = Features(
            features=matrix,
            feature_type=feature_type,
            attribute=attribute,
            origin=origin,
        )
        return feature

    def featurize_message(
        self,
        message: Message,
        featurizer_description: FeaturizerDescription,
        add_to_message: bool = True,
    ) -> Dict[Text, Featurization]:
        """Imitates the application of a featurizer.

        Note that the order of features is not guaranteed - featurizers can add
        sequence and sentenc features and also features for different attributes
        in any order.

        Args:
            message: the message to be featurized
            attribute: identifies which attribute of the given message should be
               featurized
            featurizer_description: a description of the kind of featurizer that
               should be imitated
            add_to_messages: determines whether the features should be added to the
               given message
        Returns:
            pseudo-random featurizations for each featurized attribute
        """
        # a featurizer will attempt to featurize all of the following but might
        # skip a feature if the corresponding attribute has no tokens
        potential_combinations = [
            (feature_type, dim, attribute)
            for feature_type, dim, attribute_list in [
                (
                    FEATURE_TYPE_SENTENCE,
                    featurizer_description.sentence_dim,
                    featurizer_description.sentence_attributes,
                ),
                (
                    FEATURE_TYPE_SEQUENCE,
                    featurizer_description.sequence_dim,
                    featurizer_description.sequence_attributes,
                ),
            ]
            for attribute in attribute_list
        ]
        # a featurizers can add these in any order they like, we imitate this by
        # shuffling the order:
        random.shuffle(potential_combinations)
        # generate the features
        featurizations: Dict[Text, Featurization] = dict()
        for feature_type, feature_dim, attribute in potential_combinations:
            tokens = message.get(TOKENS_NAMES[attribute], [])
            if not tokens:
                continue
            feature = self.generate_pseudo_random_feature(
                attribute=attribute,
                tokens=tokens,
                origin=featurizer_description.name,
                feature_type=feature_type,
                feature_dim=feature_dim,
                is_sparse=featurizer_description.is_sparse,
            )
            if add_to_message:
                message.features.append(feature)
            featurizations.setdefault(attribute, dict())[feature_type] = feature
        return featurizations

    def apply_featurization(self, messages: List[Message],) -> None:
        """Imitates the application of a featurization pipeline to a list of messages.

        Note that the order in which the featurizers are applied is fixed.

        Args:
            messages: some messages
        """
        for message in messages:
            for featurizer_description in self.featurizer_descriptions:
                self.featurize_message(
                    message=message,
                    featurizer_description=featurizer_description,
                    add_to_message=True,
                )

    def create_concatenated_features(
        self, messages: List[Message], attribute: Text, used_featurizers: List[Text],
    ) -> List[ConcatenatedFeaturizations]:
        """Imitates the concatenations of features of the same kind.

        For each message, all features that have been created using the
        `used_featurizers` and that have the same type and sparseness property are
        concatenated along the last dimension.

        Note that the order in which features are concatenated is *not* determined
        by the `used_featurizers` list,
        but by the order they appear in the complete definition of the featurizers.

        Args:
            messages: messages for which concatenated features should be created
            attribute: the attribute for which concatenated features should be created
            used_featurizers: the subset of the featurizers which should be used
        Returns:
            list containing, for each message, the concatenated features (i.e. one
            matrix per per type and sparseness property)
        """
        used_featurizers_ordered = [
            featurizer_description
            for featurizer_description in self.featurizer_descriptions
            if (used_featurizers is None)
            or (featurizer_description.name in used_featurizers)
        ]
        collected: List[ConcatenatedFeaturizations] = []
        for message in messages:
            message_features: ConcatenatedFeaturizations = dict()
            for is_sparse in [True, False]:
                for feature_type in [FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE]:
                    sentences = feature_type == FEATURE_TYPE_SENTENCE
                    matrices = [
                        self.generate_pseudo_random_feature(
                            attribute=attribute,
                            tokens=message.get(TOKENS_NAMES[attribute], []),
                            origin=fd.name,
                            is_sparse=fd.is_sparse,
                            feature_type=feature_type,
                            feature_dim=(
                                fd.sentence_dim if sentences else fd.sequence_dim
                            ),
                        ).features
                        # concatenated only the "used featurizers"
                        for fd in used_featurizers_ordered
                        # concatenate only those with same sparseness property
                        if fd.is_sparse == is_sparse
                        # ... if attributes the attribute was featurized
                        and attribute
                        in (
                            fd.sentence_attributes
                            if sentences
                            else fd.sequence_attributes
                        )
                        # ... if the attribute is tokenized
                        and message.get(TOKENS_NAMES[attribute], [])
                    ]
                    if matrices:
                        if is_sparse:
                            matrix = scipy.sparse.hstack(matrices)
                        else:
                            matrix = np.concatenate(matrices, axis=-1)
                        message_features[(feature_type, is_sparse)] = matrix
            collected.append(message_features)
        return collected

    @staticmethod
    def compare_with_feature_arrays(
        actual: Dict[Text, List[FeatureArray]],
        expected: List[ConcatenatedFeaturizations],
    ) -> None:
        """Compares concatenated features."""

        for feature_type in [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SENTENCE]:

            feature_array_list = actual.get(feature_type, None)

            for idx, expected_featurization in enumerate(expected):

                expected_sparse = expected_featurization.get((feature_type, True), None)
                expected_dense = expected_featurization.get((feature_type, False), None)

                if expected_sparse is None and expected_dense is None:
                    assert not feature_array_list or len(feature_array_list) == 0
                elif expected_sparse is not None and expected_dense is not None:

                    assert len(feature_array_list) == 2
                    actual_sparse, actual_dense = feature_array_list
                    assert (actual_sparse[idx] - expected_sparse).nnz == 0
                    assert np.all(np.array(actual_dense[idx]) == expected_dense)

                elif expected_dense:
                    assert len(feature_array_list) == 1
                    assert np.all(
                        np.array(feature_array_list[0][idx]) == expected_dense
                    )
                else:
                    assert len(feature_array_list) == 1
                    assert (feature_array_list[0][idx] - expected_sparse).nnz == 0
