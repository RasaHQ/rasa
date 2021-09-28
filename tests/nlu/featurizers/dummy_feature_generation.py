import random
import itertools
from dataclasses import dataclass, field
from typing import Text, List, Optional, Dict, Tuple, Union

import numpy as np
import scipy.sparse

from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.constants import (
    SENTENCE,
    SEQUENCE,
)


@dataclass
class FeaturizerDescription:
    """A description of a featurizer.

    Args:
        name: featurizer name
        seq_feat_dim: feature dimension of sequence features
        sent_feat_dim: feature dimension of sentence features
        sparse: determines whether sparse features should be created
        dense: determines whether dense features should be created
    """

    name: Text = "featurizer"
    seq_feat_dim: int = 4
    sent_feat_dim: int = 5
    sparse: bool = True
    dense: bool = True


@dataclass
class FeatureGenerator:
    """Generates pseudo-random features.

    It can generate a pseudo-random feature value and sequence length that is
    determined (and hence can be re-created) using a message attribute value,
    the attribute itself, a featurizer name, the feature type and the feature level
    (i.e. sequence/sentence level).
    Assuming the given message texts are unique, this enables us to identify
    features after they have been concatenated and stacked in various ways.

    This feature generator allows for imitating featurizers where ...
    - each featurizer may produce sparse and/or dense features
    - if a featurizer produces features of a certain kind (sparse/dense), then it
      produces sequence *and* sentence features of that kind
    - each featurizer produces at most one pair of sequence and sentence features
      for each kind (sparse/dense)
    - the dimensions of the features produced by each featurizer may differ between
      different feature types and feature levels (sequence/sentence)

    As a simplification, we assume
    - for each message and attribute, each featurizer always produces features
      with the *same* fixed sequence length
    - one can generate dense and sparse sequence and sentence features for any
      attribute

    Args:
        featurizer_descriptions: list of featurizer descriptions
        used_featurizers: list of featurizers which will be considered when
            constructing the expected concatenated features
        seq_len: a sequence length that will be used instead of the true sequence
           length of the message (hence, no need to tokenize the message)
        default_seq_feat_dim: used if no `featurizer_description` is provided
        default_sent_feat_dim:  used if no `featurizer_description` is provided
    """

    featurizer_descriptions: List[FeaturizerDescription] = field(
        default_factory=lambda: [FeaturizerDescription()]
    )
    used_featurizers: Optional[List[Text]] = None
    seq_len: int = 3
    default_seq_feat_dim: int = 4
    default_sent_feat_dim: int = 5

    @staticmethod
    def generate_feature_value(
        attribute: Text,
        attribute_value: Optional[Text],
        featurizer_name: Text,
        feature_type: Text,
    ) -> float:
        """Creates a pseudo-random value using the given parameters as seed.

        Args:
            attribute: an attribute name
            attribute_value: string representation of the attribute value
            featurizer_name: the name of a featurizer
            feature_type: either `sequence` or `sentence`

        Returns:
            a value
        """
        rng = random.Random(
            "".join(
                [
                    f"{seed_part}"
                    for seed_part in [
                        attribute,
                        attribute_value,
                        featurizer_name,
                        feature_type,
                    ]
                ]
            )
        )
        return rng.uniform(0, 100)

    @staticmethod
    def generate_seq_len(attribute: Text, attribute_value: Optional[Text],) -> int:
        rng = random.Random(
            "".join([f"{part}" for part in [attribute, attribute_value]])
        )
        return rng.uniform(1, 100)

    def generate_features(
        self, message: Message, attributes: List[Text], add_to_message: bool = True,
    ) -> Tuple[
        Dict[Text, Dict[Text, Dict[bool, Union[np.ndarray, scipy.sparse.spmatrix]]]],
        int,
    ]:
        """Generates pseudo-random features and concatenates them.

        This method will generate
        - sentence features of shape `[1, sentence_feature_dim]`
          if the attribute value is non-empty and `[0, sentence_feature_dim]` otherwise
        - sequence features of shape `[seq_len, sequence_feature_dim]`
          if the attribute value is non-empty and `[0, sequence_feature_dim]` otherwise
        where the `seq_len` is a fixed value.

        Note that if a message does not contain any value for the specificed attribute,
        a pseudo-random feature will be created nontheless (using `None` as feature
        value).
        Hence, this generator is *not* suitable for mocking features that are
        only generated if and only if a certain attribute is present in the given
        message.

        Args:
            message: a message
            add_to_message: determines whether the generated features are added to
               the message
            attributes: A list of attributes for which we want to create features.
        Returns:
            a nested mapping from attribute type, feature type and sparseness indicator
            to 2d-matrices containing a concatenation of the features that were
            generated for the `used_featurizers` (where the order of the featurizers
            is determined by the `featurizer_descriptions` and *not* the
            `used_featurizers`)
        """
        concatenations = dict()

        # per attribute and feature_type
        for attribute, feature_type in itertools.product(
            attributes, [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE]
        ):

            # (1) create features and add them to the messages if required

            # sequence length
            type_seq_len = 1 if feature_type == FEATURE_TYPE_SENTENCE else self.seq_len
            attribute_seq_len = type_seq_len if message.get(attribute) else 0
            # create and collect features per featurizer + sparseness property
            collection = dict()
            for featurizer_description in self.featurizer_descriptions:
                # feature dimension ("units")
                feat_dim = (
                    featurizer_description.sent_feat_dim
                    if feature_type == FEATURE_TYPE_SENTENCE
                    else featurizer_description.seq_feat_dim
                )
                # sparse and/or dense?
                sparse_indicators = []
                if featurizer_description.sparse:
                    sparse_indicators.append(True)
                if featurizer_description.dense:
                    sparse_indicators.append(False)
                # create and collect
                collection[featurizer_description.name] = dict()
                for sparse in sparse_indicators:
                    # create a unique value
                    value = FeatureGenerator.generate_feature_value(
                        attribute,
                        str(message.get(attribute, None)),
                        featurizer_description.name,
                        feature_type,
                    )
                    # fill a sparse/dense matrix with this value
                    if not sparse:
                        matrix = np.full(
                            shape=(attribute_seq_len, feat_dim), fill_value=value,
                        )
                    else:
                        matrix = (
                            scipy.sparse.eye(m=attribute_seq_len, n=feat_dim) * value
                        )
                    # add it to the message if required
                    if add_to_message:
                        feature = Features(
                            features=matrix,
                            feature_type=feature_type,
                            attribute=attribute,
                            origin=featurizer_description.name,
                        )
                        message.features.append(feature)
                    # ... and collect it to be able to concatenate it later
                    collection[featurizer_description.name][sparse] = matrix

            # (2) concatenate the collected features

            ordered_used_featurizers = [
                featurizer_description.name
                for featurizer_description in self.featurizer_descriptions
                if not self.used_featurizers
                or featurizer_description.name in self.used_featurizers
            ]
            concatenations.setdefault(attribute, dict()).setdefault(
                feature_type, dict()
            )
            for sparse in [True, False]:  # sparse first
                matrix_list = [
                    collection[featurizer][sparse]
                    for featurizer in ordered_used_featurizers
                    if sparse in collection[featurizer]
                ]
                if not matrix_list:
                    continue
                if sparse:
                    concat_matrix = scipy.sparse.hstack(matrix_list)
                else:
                    concat_matrix = np.concatenate(matrix_list, axis=-1)
                concatenations[attribute][feature_type][sparse] = concat_matrix

        return concatenations

    def generate_features_for_all_messages(
        self,
        messages: List[Message],
        attributes: List[Text],
        add_to_messages: bool = True,
    ) -> Dict[
        Text,
        Dict[Text, Dict[bool, Union[List[np.ndarray], List[scipy.sparse.spmatrix]]]],
    ]:
        """Generates pseudo-random dense features and concatenates them.

        For more details, see `generate_features`.

        Args:
            messages: a list of messages
            add_to_messages: determines whether the generated features are added to
               the messages
            attributes: A list of attributes for which we want to create features.
        Returns:
            a nested mapping from attribute type, feature type and sparseness indicator
            to 2d-matrices containing a list that contains, for each given message,
            a concatenation of the features that were generated for the
            `self.used_featurizers` and the respective message.
        """
        concatenated_features = {
            attribute: {
                feature_type: dict()  # mapping is_sparse (bool) to list of matrices
                for feature_type in [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE]
            }
            for attribute in attributes
        }
        for message in messages:
            message_features = self.generate_features(
                message=message, add_to_message=add_to_messages, attributes=attributes,
            )
            for attribute in attributes:
                for feature_type in [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE]:
                    sparse_and_dense = message_features[attribute][feature_type]
                    for is_sparse, matrix in sparse_and_dense.items():
                        concatenated_features[attribute][feature_type].setdefault(
                            is_sparse, list()
                        ).append(matrix)
        return concatenated_features

    @classmethod
    def compare_features_of_same_type_and_sparseness(
        cls,
        actual: Dict[Text, Union[List[np.ndarray], List[scipy.sparse.spmatrix]]],
        expected: Dict[Text, Dict[bool, Union[np.ndarray, scipy.sparse.spmatrix]]],
        indices_of_expected: Optional[List[int]] = None,
    ) -> None:
        """Compares mappings from feature type and sparseness indicator to features.

        Note that this comparison allows for feature types to be missing,
        while it always expects 'sparse' *and* 'dense' features to be populated.

        Args:
            actual: mapping feature type to a list containing a sparse scipy matrix
               and/or a dense numpy array, with the sparse matrix first
            expected: mapping feature type and sparseness indicator (`True`/`False`)
               to a dense numpy array or a scipy sparse matrix, respectively
            indices_of_expected: indices describing which messages from `expected`
               should be considered for the comparison
        """
        differences_found = []
        for feature_type in [SENTENCE, SEQUENCE]:
            if feature_type not in expected:
                assert feature_type not in actual
                continue
            assert len(actual[feature_type]) == len(expected[feature_type])
            for is_sparse in expected[feature_type]:
                if len(expected[feature_type]) == 2:
                    sparse_idx = not is_sparse  # i.e. sparse 0, dense 1
                else:
                    sparse_idx = 0
                feature_array = actual[feature_type][sparse_idx]
                expected_feature_array = expected[feature_type][is_sparse]
                relevant_indices = indices_of_expected or np.arange(
                    len(expected_feature_array)
                )
                assert len(feature_array) == len(relevant_indices)
                for idx_actual, idx_expected in enumerate(relevant_indices):
                    actual_matrix = feature_array[idx_actual]
                    expected_matrix = expected_feature_array[idx_expected]
                    if is_sparse:
                        actual_matrix = actual_matrix.todense()
                        expected_matrix = expected_matrix.todense()
                    else:
                        # because it's a feature array
                        actual_matrix = np.array(actual_matrix)

                    if actual_matrix.shape != expected_matrix.shape or not np.all(
                        actual_matrix == expected_matrix
                    ):
                        message = (
                            f"difference detected for "
                            f"sparse = {is_sparse}, "
                            f"feature_type = {feature_type}, "
                            f"idx_actual = {idx_actual} and "
                            f"idx_expected = {idx_expected}: "
                            f"shapes = {actual_matrix.shape} vs. "
                            f"{expected_matrix.shape} "
                            f"values = {actual_matrix} vs. {expected_matrix}"
                        )
                        differences_found.append(message)
        assert not differences_found, "\n" + "\n".join(differences_found)
