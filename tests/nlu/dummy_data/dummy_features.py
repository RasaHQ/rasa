import random
from dataclasses import dataclass, field
from typing import OrderedDict, Set, Text, List, Optional, TypedDict

import numpy as np
import scipy.sparse
from rasa.nlu.constants import TOKENS_NAMES

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
        dimension: feature dimension of sequence and sentence features
        is_sparse: determines whether it produces sparse features
    """

    name: Text = "featurizer"
    dimension: int = 4
    # TODO: do we have featurizer where sentence and sequence feature dimensions differ?
    is_sparse: bool = True


# Each featurizer should produce sequence and sentence features. However, if the
# attribute to be featurized is not present, then no features are generated.
# Moreover, not all featurizers always produce sentence features.
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
       always produces sequence *and* sentence features of that kind and ...
    3. ... those features always have the same (last) dimension (also denoted "units"
       in other parts of the code base)
    4. each featurizer can produces features for an attribute if and only if that
       attribute has been tokenized (i.e. the list of tokens is not empty) and will
       produce sequence features with a sequence length that is equal to the number
       of tokens
    5. featurizers are always applied exactly once to a message - in the exact same
       sequential order

    Note that assumption 2. is violated e.g. by `LexicalSyntacticFeaturizer` and the
    `CountVectorizer` (when applied to attributes that are not "dense featurizable").
    To run tests with these assumptions, features need to be removed from the
    dummy data by hand.

    Note that 3. and 6. is are essential assumptions in the current code base
    regarding the data prepartion for models and regarding the models themselves
    which expect to be able to concatenate sentence and sequence features.

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

    def create_features(
        self,
        message: Message,
        attribute: Text,
        featurizer_description: FeaturizerDescription,
        add_to_message: bool = True,
        skip_sentence_features: bool = False,
    ) -> Featurization:
        """Imitates the application of a featurizer to a specific message attribute.

        Args:
            ...
        """
        tokens = message.get(TOKENS_NAMES[attribute], [])
        if not tokens:
            return dict()
        featurization = dict()
        # order of sequence and sentence features is not guaranteed - featurizers
        # can add them in any order
        feature_types = [FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE]
        random.shuffle(feature_types)
        for feature_type in feature_types:
            # skip sentence features if requested
            if feature_type == FEATURE_TYPE_SENTENCE and skip_sentence_features:
                continue
            # shape
            seq_len = 1 if feature_type == FEATURE_TYPE_SENTENCE else len(tokens)
            feat_dim = featurizer_description.dimension
            shape = (seq_len, feat_dim)
            # create a unique seed value and initialise an rng
            seed = self.generate_pseudo_random_value(
                [
                    attribute,
                    message.get(attribute, None),
                    featurizer_description.name,
                    feature_type,
                ]
            )
            rng = np.random.default_rng(seed)
            # fill a sparse/dense matrix using the rng
            if not featurizer_description.is_sparse:
                matrix = rng.random(shape)
            else:
                matrix = scipy.sparse.random(m=shape[0], n=shape[1], random_state=rng)
            # add it to the message if required
            feature = Features(
                features=matrix,
                feature_type=feature_type,
                attribute=attribute,
                origin=featurizer_description.name,
            )
            if add_to_message:
                message.features.append(feature)
            featurization[feature_type] = feature
        return featurization

    def featurize_messages(
        self,
        messages: List[Message],
        attributes: List[Text],
        attributes_without_sentence_features: Set[Text],
    ) -> None:
        """Imitates the application of a featurization pipeline to a list of messages.

        Args:
            messages: some messages
            attributes: a list of attributes for which we want to create features.
        """
        for message in messages:
            for attribute in attributes:
                # the order in which the featurizers are applied is fixed
                skip_sentence_features = (
                    attribute in attributes_without_sentence_features
                )
                for featurizer_description in self.featurizer_descriptions:
                    self.create_features(
                        message=message,
                        attribute=attribute,
                        featurizer_description=featurizer_description,
                        add_to_message=True,
                        skip_sentence_features=skip_sentence_features,
                    )

    def create_concatenated_features(
        self,
        messages: List[Message],
        attribute: Text,
        used_featurizers: List[Text],
        skip_sentence_features: bool,
    ) -> List[ConcatenatedFeaturizations]:
        """Imitates the concatenations of features of the same kind.

        For each message, all features that have been created using the
        `used_featurizers` and that have the same type and sparseness property are
        concatenated along the last dimension.
        Their order is determined *not* determined by the `used_featurizers` list
        but by the order they appear in the complete definition of the featurizers.

        Args:
            list containing, for each message, the concatenated features (i.e. one
            matrix per per type and sparseness property)
        """

        concatenation_order = [
            featurizer_description
            for featurizer_description in self.featurizer_descriptions
            if (used_featurizers is None)
            or (featurizer_description.name in used_featurizers)
        ]

        collected: List[ConcatenatedFeaturizations] = []
        for message in messages:
            message_features: ConcatenatedFeaturizations = dict()
            for is_sparse in [True, False]:
                featurizations = [
                    self.create_features(
                        message=message,
                        attribute=attribute,
                        featurizer_description=featurizer,
                        add_to_message=False,
                        skip_sentence_features=skip_sentence_features,
                    )  # i.e. sequence and sentence features
                    for featurizer in concatenation_order
                    if featurizer.is_sparse == is_sparse
                ]
                for feature_type in [FEATURE_TYPE_SEQUENCE, FEATURE_TYPE_SENTENCE]:
                    matrices = [
                        featurization[feature_type].features
                        for featurization in featurizations
                        if feature_type in featurization
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
        actual: OrderedDict[Text, List[FeatureArray]],
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
