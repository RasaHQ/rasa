import tensorflow as tf
import numpy as np
from typing import Text, List, Dict, Any, Union, Optional, Tuple, Callable

from rasa.shared.nlu.constants import TEXT
from rasa.utils.tensorflow.model_data import FeatureSignature
from rasa.utils.tensorflow.constants import (
    REGULARIZATION_CONSTANT,
    CONNECTION_DENSITY,
    NUM_TRANSFORMER_LAYERS,
    TRANSFORMER_SIZE,
    NUM_HEADS,
    UNIDIRECTIONAL_ENCODER,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    MASKED_LM,
    HIDDEN_LAYERS_SIZES,
    DROP_RATE,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    DENSE_DIMENSION,
    CONCAT_DIMENSION,
    DROP_RATE_ATTENTION,
    SEQUENCE,
    SENTENCE,
)
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.exceptions import TFLayerConfigException
from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.nlu.constants import DEFAULT_TRANSFORMER_SIZE


class RasaCustomLayer(tf.keras.layers.Layer):
    """Parent class for all classes in `rasa_layers.py`.

    Allows a shared implementation for adjusting `DenseForSparse`
    layers during incremental training.

    During fine-tuning, sparse feature sizes might change due to addition of new data.
    If this happens, we need to adjust our `DenseForSparse` layers to a new size.
    `ConcatenateSparseDenseFeatures`, `RasaSequenceLayer` and
    `RasaFeatureCombiningLayer` all inherit from `RasaCustomLayer` and thus can
    change their own `DenseForSparse` layers if it's needed.
    """

    def adjust_sparse_layers_for_incremental_training(
        self,
        new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]],
        reg_lambda: float,
    ) -> None:
        """Finds and adjusts `DenseForSparse` layers during incremental training.

        Recursively looks through the layers until it finds all the `DenseForSparse`
        ones and adjusts those which have their sparse feature sizes increased.

        This function heavily relies on the name of `DenseForSparse` layer being
        in the following format - f"sparse_to_dense.{attribute}_{feature_type}" -
        in order to correctly extract the attribute and feature type.

        New and old sparse feature sizes could look like this:
        {TEXT: {FEATURE_TYPE_SEQUENCE: [4, 24, 128], FEATURE_TYPE_SENTENCE: [4, 128]}}

        Args:
            new_sparse_feature_sizes: sizes of current sparse features.
            old_sparse_feature_sizes: sizes of sparse features the model was
                                      previously trained on.
            reg_lambda: regularization constant.
        """
        for name, layer in self._tf_layers.items():
            if isinstance(layer, RasaCustomLayer):
                layer.adjust_sparse_layers_for_incremental_training(
                    new_sparse_feature_sizes=new_sparse_feature_sizes,
                    old_sparse_feature_sizes=old_sparse_feature_sizes,
                    reg_lambda=reg_lambda,
                )
            elif isinstance(layer, layers.DenseForSparse):
                attribute = layer.get_attribute()
                feature_type = layer.get_feature_type()
                if (
                    attribute in new_sparse_feature_sizes
                    and feature_type in new_sparse_feature_sizes[attribute]
                ):
                    new_feature_sizes = new_sparse_feature_sizes[attribute][
                        feature_type
                    ]
                    old_feature_sizes = old_sparse_feature_sizes[attribute][
                        feature_type
                    ]
                    if sum(new_feature_sizes) > sum(old_feature_sizes):
                        self._tf_layers[name] = self._replace_dense_for_sparse_layer(
                            layer_to_replace=layer,
                            new_sparse_feature_sizes=new_feature_sizes,
                            old_sparse_feature_sizes=old_feature_sizes,
                            attribute=attribute,
                            feature_type=feature_type,
                            reg_lambda=reg_lambda,
                        )

    @staticmethod
    def _replace_dense_for_sparse_layer(
        layer_to_replace: layers.DenseForSparse,
        new_sparse_feature_sizes: List[int],
        old_sparse_feature_sizes: List[int],
        attribute: Text,
        feature_type: Text,
        reg_lambda: float,
    ) -> layers.DenseForSparse:
        """Replaces a `DenseForSparse` layer with a new one.

        Replaces an existing `DenseForSparse` layer with a new one
        in order to adapt it to incremental training.

        Args:
            layer_to_replace: a `DenseForSparse` layer that is used to create a new one.
            new_sparse_feature_sizes: sizes of sparse features that will be
                                      the input of the layer.
            old_sparse_feature_sizes: sizes of sparse features that used to be
                                      the input of the layer.
            attribute: an attribute of the data fed to the layer.
            feature_type: a feature type of the data fed to the layer.
            reg_lambda: regularization constant.

        Returns:
            New `DenseForSparse` layer.
        """
        kernel = layer_to_replace.get_kernel().numpy()
        bias = layer_to_replace.get_bias()
        if bias is not None:
            bias = bias.numpy()
        units = layer_to_replace.get_units()
        # split kernel by feature sizes to update the layer accordingly
        kernel_splits = []
        splitting_index = 0
        for size in old_sparse_feature_sizes:
            kernel_splits.append(kernel[splitting_index : splitting_index + size, :])
            splitting_index += size
        additional_sizes = [
            new_size - old_size
            for new_size, old_size in zip(
                new_sparse_feature_sizes, old_sparse_feature_sizes
            )
        ]
        std, mean = np.std(kernel), np.mean(kernel)
        additional_weights = [
            np.random.normal(mean, std, size=(num_rows, units)).astype(np.float32)
            for num_rows in additional_sizes
        ]
        merged_weights = [
            np.vstack((existing, new))
            for existing, new in zip(kernel_splits, additional_weights)
        ]
        # stack each merged weight to form a new weight tensor
        new_weights = np.vstack(merged_weights)
        kernel_init = tf.constant_initializer(new_weights)
        bias_init = tf.constant_initializer(bias) if bias is not None else None
        new_layer = layers.DenseForSparse(
            name=f"sparse_to_dense.{attribute}_{feature_type}",
            reg_lambda=reg_lambda,
            units=units,
            use_bias=bias is not None,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
        return new_layer


class ConcatenateSparseDenseFeatures(RasaCustomLayer):
    """Combines multiple sparse and dense feature tensors into one dense tensor.

    This layer combines features from various featurisers into a single feature array
    per input example. All features must be of the same feature type, i.e. sentence-
    level or sequence-level (token-level).

    The layer combines a given list of tensors (whether sparse or dense) by:
    1. converting sparse tensors into dense ones
    2. optionally, applying dropout to sparse tensors before and/or after the conversion
    3. concatenating all tensors along the last dimension

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        feature_type: Feature type to be processed -- `sequence` or `sentence`.
        feature_type_signature: A list of signatures for the given attribute and feature
            type.
        config: A model config for correctly parametrising the layer.

    Input shape:
        Tuple containing one list of N-D tensors, each with shape: `(batch_size, ...,
        input_dim)`.
        All dense tensors must have the same shape, except possibly the last dimension.
        All sparse tensors must have the same shape, including the last dimension.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)` where `units` is the sum of
        the last dimension sizes across all input tensors, with sparse tensors instead
        contributing `config[DENSE_DIMENSION][attribute]` units each.

    Raises:
        A `TFLayerConfigException` if no feature signatures are provided.

    Attributes:
        output_units: The last dimension size of the layer's output.
    """

    SPARSE_DROPOUT = "sparse_dropout"
    SPARSE_TO_DENSE = "sparse_to_dense"
    DENSE_DROPOUT = "dense_dropout"

    def __init__(
        self,
        attribute: Text,
        feature_type: Text,
        feature_type_signature: List[FeatureSignature],
        config: Dict[Text, Any],
    ) -> None:
        """Creates a new `ConcatenateSparseDenseFeatures` object."""
        if not feature_type_signature:
            raise TFLayerConfigException(
                "The feature type signature must contain some feature signatures."
            )

        super().__init__(
            name=f"concatenate_sparse_dense_features_{attribute}_{feature_type}"
        )

        self._check_sparse_input_units(feature_type_signature)

        self.output_units = self._calculate_output_units(
            attribute, feature_type_signature, config
        )

        # Prepare dropout and sparse-to-dense layers if any sparse tensors are expected
        self._tf_layers: Dict[Text, tf.keras.layers.Layer] = {}
        if any([signature.is_sparse for signature in feature_type_signature]):
            self._prepare_layers_for_sparse_tensors(attribute, feature_type, config)

    def _check_sparse_input_units(
        self, feature_type_signature: List[FeatureSignature]
    ) -> None:
        """Checks that all sparse features have the same last dimension size."""
        sparse_units = [
            feature_sig.units
            for feature_sig in feature_type_signature
            if feature_sig.is_sparse
        ]
        if len(set(sparse_units)) > 1:
            raise TFLayerConfigException(
                f"All sparse features must have the same last dimension size but found "
                f"different sizes: {set(sparse_units)}."
            )

    def _prepare_layers_for_sparse_tensors(
        self, attribute: Text, feature_type: Text, config: Dict[Text, Any]
    ) -> None:
        """Sets up sparse tensor pre-processing before combining with dense ones."""
        # For optionally applying dropout to sparse tensors
        if config[SPARSE_INPUT_DROPOUT]:
            self._tf_layers[self.SPARSE_DROPOUT] = layers.SparseDropout(
                rate=config[DROP_RATE]
            )

        # For converting sparse tensors to dense
        self._tf_layers[self.SPARSE_TO_DENSE] = layers.DenseForSparse(
            name=f"sparse_to_dense.{attribute}_{feature_type}",
            units=config[DENSE_DIMENSION][attribute],
            reg_lambda=config[REGULARIZATION_CONSTANT],
        )

        # For optionally apply dropout to sparse tensors after they're converted to
        # dense tensors.
        if config[DENSE_INPUT_DROPOUT]:
            self._tf_layers[self.DENSE_DROPOUT] = tf.keras.layers.Dropout(
                rate=config[DROP_RATE]
            )

    @staticmethod
    def _calculate_output_units(
        attribute: Text,
        feature_type_signature: List[FeatureSignature],
        config: Dict[Text, Any],
    ) -> int:
        """Determines the output units from the provided feature signatures.

        Sparse features will be turned into dense ones, hence they each contribute with
        their future dense number of units.
        """
        return sum(
            [
                config[DENSE_DIMENSION][attribute]
                if signature.is_sparse
                else signature.units
                for signature in feature_type_signature
            ]
        )

    def _process_sparse_feature(
        self, feature: tf.SparseTensor, training: bool
    ) -> tf.Tensor:
        """Turns sparse tensor into dense, possibly adds dropout before and/or after."""
        if self.SPARSE_DROPOUT in self._tf_layers:
            feature = self._tf_layers[self.SPARSE_DROPOUT](feature, training)

        feature = self._tf_layers[self.SPARSE_TO_DENSE](feature)

        if self.DENSE_DROPOUT in self._tf_layers:
            feature = self._tf_layers[self.DENSE_DROPOUT](feature, training)

        return feature

    def call(
        self,
        inputs: Tuple[List[Union[tf.Tensor, tf.SparseTensor]]],
        training: bool = False,
    ) -> tf.Tensor:
        """Combines sparse and dense feature tensors into one tensor.

        Arguments:
            inputs: Contains the input tensors, all of the same rank.
            training: A flag indicating whether the layer should behave in training mode
                (applying dropout to sparse tensors if applicable) or in inference mode
                (not applying dropout).

        Returns:
            Single tensor with all input tensors combined along the last dimension.
        """
        features = inputs[0]

        dense_features = []
        for f in features:
            if isinstance(f, tf.SparseTensor):
                f = self._process_sparse_feature(f, training)
            dense_features.append(f)

        # Now that all features are made dense, concatenate them along the last (units)
        # dimension.
        return tf.concat(dense_features, axis=-1)


class RasaFeatureCombiningLayer(RasaCustomLayer):
    """Combines multiple dense or sparse feature tensors into one.

    This layer combines features by following these steps:
    1. Apply a `ConcatenateSparseDenseFeatures` layer separately to sequence- and
        sentence-level features, yielding two tensors (one for each feature type).
    2. Concatenate the sequence- and sentence-level tensors along the sequence dimension
        by appending sentence-level features at the first available token position after
        the sequence-level (token-level) features.

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        attribute_signature: A dictionary containing two lists of feature signatures,
            one for each feature type (`sequence` or `sentence`) of the given attribute.
        config: A model config used for correctly parameterising the layer and the
            `ConcatenateSparseDenseFeatures` layer it uses internally.

    Input shape:
        Tuple of three input tensors:
            sequence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
                different for sparse vs dense tensors. See the input shape of
                `ConcatenateSparseDenseFeatures` for more information.
            sentence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, 1, input_dim)` where `input_dim` can be different for
                sparse vs dense tensors, and can differ from that in
                `sequence_features`. See the input shape of
                `ConcatenateSparseDenseFeatures` for more information.
            sequence_feature_lengths: Dense tensor of shape `(batch_size, )`.

    Output shape:
        combined_features: A 3-D tensor with shape `(batch_size, sequence_length,
            units)` where `units` is  completely  determined by the internally applied
            `ConcatenateSparseDenseFeatures` layer and `sequence_length` is the combined
            length of sequence- and sentence-level features: `max_seq_length + 1` if
            both feature types are present, `max_seq_length` if only sequence-level
            features are present, and 1 if only sentence-level features are present).
        mask_combined_sequence_sentence: A 3-D tensor with shape
            `(batch_size, sequence_length, 1)`.

    Raises:
        A `TFLayerConfigException` if no feature signatures are provided.

    Attributes:
        output_units: The last dimension size of the layer's `combined_features` output.
    """

    def __init__(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        """Creates a new `RasaFeatureCombiningLayer` object."""
        if not attribute_signature or not (
            attribute_signature.get(SENTENCE, [])
            or attribute_signature.get(SEQUENCE, [])
        ):
            raise TFLayerConfigException(
                "The attribute signature must contain some feature signatures."
            )

        super().__init__(name=f"rasa_feature_combining_layer_{attribute}")

        self._tf_layers: Dict[Text, tf.keras.layers.Layer] = {}

        # Prepare sparse-dense combining layers for each present feature type
        self._feature_types_present = self._get_present_feature_types(
            attribute_signature
        )
        self._prepare_sparse_dense_concat_layers(attribute, attribute_signature, config)

        # Prepare components for combining sequence- and sentence-level features
        self._prepare_sequence_sentence_concat(attribute, config)

        self.output_units = self._calculate_output_units(attribute, config)

    @staticmethod
    def _get_present_feature_types(
        attribute_signature: Dict[Text, List[FeatureSignature]]
    ) -> Dict[Text, bool]:
        """Determines feature types that are present.

        Knowing which feature types are present is important because many downstream
        operations depend on it, e.g. combining sequence- and sentence-level features
        is only done if both feature types are present.
        """
        return {
            feature_type: (
                feature_type in attribute_signature
                and len(attribute_signature[feature_type]) > 0
            )
            for feature_type in [SEQUENCE, SENTENCE]
        }

    def _prepare_sparse_dense_concat_layers(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        """Prepares sparse-dense combining layers for all present feature types."""
        for feature_type, present in self._feature_types_present.items():
            if not present:
                continue
            self._tf_layers[
                f"sparse_dense.{feature_type}"
            ] = ConcatenateSparseDenseFeatures(
                attribute=attribute,
                feature_type=feature_type,
                feature_type_signature=attribute_signature[feature_type],
                config=config,
            )

    def _prepare_sequence_sentence_concat(
        self, attribute: Text, config: Dict[Text, Any]
    ) -> None:
        """Sets up combining sentence- and sequence-level features (if needed).

        This boils down to preparing for unifying the units of the sequence- and
        sentence-level features if they differ -- the same number of units is required
        for combining the features.
        """
        if (
            self._feature_types_present[SEQUENCE]
            and self._feature_types_present[SENTENCE]
        ):
            # The output units of this layer will be based on the output sizes of the
            # sparse+dense combining layers that are internally applied to all features.
            sequence_units = self._tf_layers[f"sparse_dense.{SEQUENCE}"].output_units
            sentence_units = self._tf_layers[f"sparse_dense.{SENTENCE}"].output_units

            # Last dimension needs to be unified if sequence- and sentence-level
            # features have different sizes, e.g. due to being produced by different
            # featurizers.
            if sequence_units != sentence_units:
                for feature_type in [SEQUENCE, SENTENCE]:
                    self._tf_layers[
                        f"unify_dims_before_seq_sent_concat.{feature_type}"
                    ] = layers.Ffnn(
                        layer_name_suffix=f"unify_dims.{attribute}_{feature_type}",
                        layer_sizes=[config[CONCAT_DIMENSION][attribute]],
                        dropout_rate=config[DROP_RATE],
                        reg_lambda=config[REGULARIZATION_CONSTANT],
                        density=config[CONNECTION_DENSITY],
                    )

    def _calculate_output_units(self, attribute: Text, config: Dict[Text, Any]) -> int:
        """Calculates the number of output units for this layer class.

        The number depends mainly on whether dimension unification is used or not.
        """
        # If dimension unification is used, output units are determined by the unifying
        # layers.
        if (
            f"unify_dims_before_seq_sent_concat.{SEQUENCE}" in self._tf_layers
            or f"unify_dims_before_seq_sent_concat.{SENTENCE}" in self._tf_layers
        ):
            return config[CONCAT_DIMENSION][attribute]
        # Without dimension unification, the units from the underlying sparse_dense
        # layers are carried over and should be the same for sequence-level features
        # (if present) as for sentence-level features.
        elif self._feature_types_present[SEQUENCE]:
            return self._tf_layers[f"sparse_dense.{SEQUENCE}"].output_units
        return self._tf_layers[f"sparse_dense.{SENTENCE}"].output_units

    def _concat_sequence_sentence_features(
        self,
        sequence_tensor: tf.Tensor,
        sentence_tensor: tf.Tensor,
        mask_combined_sequence_sentence: tf.Tensor,
    ) -> tf.Tensor:
        """Concatenates sequence- & sentence-level features along sequence dimension."""
        # If needed, pass both feature types through a dense layer to bring them to the
        # same shape.
        if f"unify_dims_before_seq_sent_concat.{SEQUENCE}" in self._tf_layers:
            sequence_tensor = self._tf_layers[
                f"unify_dims_before_seq_sent_concat.{SEQUENCE}"
            ](sequence_tensor)
        if f"unify_dims_before_seq_sent_concat.{SENTENCE}" in self._tf_layers:
            sentence_tensor = self._tf_layers[
                f"unify_dims_before_seq_sent_concat.{SENTENCE}"
            ](sentence_tensor)

        # mask_combined_sequence_sentence has for each input example a sequence of 1s of
        # the length seq_length+1, where seq_length is the number of real tokens. The
        # rest is 0s which form a padding up to the max. sequence length + 1 (max.
        # number of real tokens + 1). Here the mask is turned into a mask that has 0s
        # everywhere and 1 only at the immediate next position after the last real
        # token's position for a given input example. Example (batch size = 2, sequence
        # lengths = [1, 2]):
        # [[[1], [0], [0]],     ___\   [[[0], [1], [0]],
        #  [[1], [1], [0]]]        /    [[0], [0], [1]]]
        sentence_feature_positions_mask = (
            mask_combined_sequence_sentence
            * tf.math.cumprod(
                1 - mask_combined_sequence_sentence,
                axis=1,
                exclusive=True,
                reverse=True,
            )
        )

        # The new mask is used to distribute the sentence features at the sequence
        # positions marked by 1s. The sentence features' dimensionality effectively
        # changes from `(batch_size, 1, feature_dim)` to `(batch_size, max_seq_length+1,
        # feature_dim)`, but the array is sparse, with real features present only at
        # positions determined by 1s in the mask.
        sentence_tensor = sentence_feature_positions_mask * sentence_tensor

        # Padding of sequence-level features is increased by 1 in the sequence
        # dimension to match the shape of modified sentence-level features.
        sequence_tensor = tf.pad(sequence_tensor, [[0, 0], [0, 1], [0, 0]])

        # Sequence- and sentence-level features effectively get concatenated by
        # summing the two padded feature arrays like this (batch size = 1):
        # [[seq1, seq2, seq3, 0, 0]] + [[0, 0, 0, sent1, 0]] =
        # = [[seq1, seq2, seq3, sent1, 0]]
        return sequence_tensor + sentence_tensor

    def _combine_sequence_level_features(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask_sequence: tf.Tensor,
        training: bool,
    ) -> Optional[tf.Tensor]:
        """Processes & combines sequence-level features if any are present."""
        if self._feature_types_present[SEQUENCE]:
            sequence_features_combined = self._tf_layers[f"sparse_dense.{SEQUENCE}"](
                (sequence_features,), training=training
            )

            # Apply mask which has 1s at positions of real tokens and 0s at all padded
            # token positions. This is needed because the sparse+dense combining layer
            # might've turned some fake (padded) features (i.e. 0s) into non-zero
            # numbers and we want those to become zeros again.
            # This step isn't needed for sentence-level features because those are never
            # padded -- the effective sequence length in their case is always 1.
            return sequence_features_combined * mask_sequence

        return None

    def _combine_sentence_level_features(
        self,
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sequence_feature_lengths: tf.Tensor,
        training: bool,
    ) -> Tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
        """Processes & combines sentence-level features if any are present."""
        if self._feature_types_present[SENTENCE]:
            sentence_features_combined = self._tf_layers[f"sparse_dense.{SENTENCE}"](
                (sentence_features,), training=training
            )
            # Sentence-level features have sequence dimension of length 1, add it to
            # sequence-level feature lengths.
            combined_sequence_sentence_feature_lengths = sequence_feature_lengths + 1

        else:
            sentence_features_combined = None

            # Without sentence-level features, the feature sequence lengths are
            # completely determined by sequence-level features.
            combined_sequence_sentence_feature_lengths = sequence_feature_lengths

        return sentence_features_combined, combined_sequence_sentence_feature_lengths

    def call(
        self,
        inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
        ],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Combines multiple 3-D dense/sparse feature tensors into one.

        Arguments:
            inputs: Tuple containing:
                sequence_features: Dense or sparse tensors representing different
                    token-level features.
                sentence_features: Dense or sparse tensors representing sentence-level
                    features.
                sequence_feature_lengths: A tensor containing the real sequence length
                    (the number of real -- not padding -- tokens) for each example in
                    the batch.
            training: A flag indicating whether the layer should behave in training mode
                (applying dropout to sparse tensors if applicable) or in inference mode
                (not applying dropout).

        Returns:
            combined features: A tensor containing all the features combined.
            mask_combined_sequence_sentence: A binary mask with 1s in place of real
                features in the combined feature tensor, and 0s in padded positions with
                fake features.
        """
        sequence_features = inputs[0]
        sentence_features = inputs[1]
        sequence_feature_lengths = inputs[2]

        # This mask is specifically for sequence-level features.
        mask_sequence = compute_mask(sequence_feature_lengths)

        sequence_features_combined = self._combine_sequence_level_features(
            sequence_features, mask_sequence, training
        )

        (
            sentence_features_combined,
            combined_sequence_sentence_feature_lengths,
        ) = self._combine_sentence_level_features(
            sentence_features, sequence_feature_lengths, training
        )

        mask_combined_sequence_sentence = compute_mask(
            combined_sequence_sentence_feature_lengths
        )

        # If both feature types are present, combine them. Otherwise, just the present
        # feature type will be returned.
        if (
            sequence_features_combined is not None
            and sentence_features_combined is not None
        ):
            features_to_return = self._concat_sequence_sentence_features(
                sequence_features_combined,
                sentence_features_combined,
                mask_combined_sequence_sentence,
            )
        elif sequence_features_combined is not None:
            features_to_return = sequence_features_combined
        else:
            features_to_return = sentence_features_combined

        return features_to_return, mask_combined_sequence_sentence


class RasaSequenceLayer(RasaCustomLayer):
    """Creates an embedding from all features for a sequence attribute; facilitates MLM.

    This layer combines all features for an attribute and embeds them using a
    transformer, optionally doing masked language modeling. The layer is meant only for
    attributes with sequence-level features, such as `text`, `response` and
    `action_text`.

    Internally, this layer applies the following steps:
    1. Combine features using `RasaFeatureCombiningLayer`.
    2. Apply a dense layer(s) to the combined features.
    3. Optionally, and only during training for the `text` attribute, apply masking to
        the features and create further helper variables for masked language modeling.
    4. Embed the features using a transformer, effectively reducing variable-length
        sequences of features to fixed-size embeddings.

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        attribute_signature: A dictionary containing two lists of feature signatures,
            one for each feature type (`sentence` or `sequence`) of the given attribute.
        config: A model config used for correctly parameterising the underlying layers.

    Input shape:
        Tuple of three input tensors:
            sequence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
                different for sparse vs dense tensors. See the input shape of
                `ConcatenateSparseDenseFeatures` for more information.
            sentence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, 1, input_dim)` where `input_dim` can be different for
                sparse vs dense tensors, and can differ from that in
                `sequence_features`. See the input shape of
                `ConcatenateSparseDenseFeatures` for more information.
            sequence_feature_lengths: Dense tensor of shape `(batch_size, )`.

    Output shape:
        outputs: `(batch_size, seq_length, units)` where `units` matches the underlying
            transformer's output size (if present), otherwise it matches the output size
            of the `Ffnn` block applied to the combined features, or it's the output
            size of the underlying `RasaFeatureCombiningLayer` if the `Ffnn` block has 0
            layers. `seq_length` is the sum of the sequence dimension
            sizes of sequence- and sentence-level features (for details, see the output
            shape of `RasaFeatureCombiningLayer`). If both feature types are present,
            then `seq_length` will be 1 + the length of the longest sequence of real
            tokens across all examples in the given batch.
        seq_sent_features: `(batch_size, seq_length, hidden_dim)`, where `hidden_dim` is
            the output size of the underlying `Ffnn` block, or the output size of the
            underlying `RasaFeatureCombiningLayer` if the `Ffnn` block has 0 layers.
        mask_combined_sequence_sentence: `(batch_size, seq_length, 1)`
        token_ids: `(batch_size, seq_length, id_dim)`. `id_dim` is 2 when no dense
            sequence-level features are present. Otherwise, it's arbitrarily chosen to
            match the last dimension size of the first dense sequence-level feature in
            the input list of features.
        mlm_boolean_mask: `(batch_size, seq_length, 1)`, empty tensor if not doing MLM.
        attention_weights: `(transformer_layers, batch_size, num_transformer_heads,
            seq_length, seq_length)`, empty tensor if the transformer has 0 layers.

    Raises:
        A `TFLayerConfigException` if no feature signatures for sequence-level features
            are provided.

    Attributes:
        output_units: The last dimension size of the layer's first output (`outputs`).
    """

    FEATURE_COMBINING = "feature_combining"
    FFNN = "ffnn"
    TRANSFORMER = "transformer"
    MLM_INPUT_MASK = "mlm_input_mask"
    SPARSE_TO_DENSE_FOR_TOKEN_IDS = "sparse_to_dense_for_token_ids"

    def __init__(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        """Creates a new `RasaSequenceLayer` object."""
        if not attribute_signature or not attribute_signature.get(SEQUENCE, []):
            raise TFLayerConfigException(
                "The attribute signature must contain some sequence-level feature"
                "signatures but none were found."
            )

        super().__init__(name=f"rasa_sequence_layer_{attribute}")

        self._tf_layers: Dict[Text, Any] = {
            self.FEATURE_COMBINING: RasaFeatureCombiningLayer(
                attribute, attribute_signature, config
            ),
            self.FFNN: layers.Ffnn(
                config[HIDDEN_LAYERS_SIZES][attribute],
                config[DROP_RATE],
                config[REGULARIZATION_CONSTANT],
                config[CONNECTION_DENSITY],
                layer_name_suffix=attribute,
            ),
        }

        self._enables_mlm = False
        # Note: Within TED, masked language modeling becomes just input dropout,
        # since there is no loss term associated with predicting the masked tokens.
        self._prepare_masked_language_modeling(attribute, attribute_signature, config)

        transformer_layers, transformer_units = self._prepare_transformer(
            attribute, config
        )
        self._has_transformer = transformer_layers > 0

        self.output_units = self._calculate_output_units(
            attribute, transformer_layers, transformer_units, config
        )

    @staticmethod
    def _get_transformer_dimensions(
        attribute: Text, config: Dict[Text, Any]
    ) -> Tuple[int, int]:
        """Determines # of transformer layers & output size from the model config.

        The config can contain these directly (same for all attributes) or specified
        separately for each attribute.
        If a transformer is used (e.i. if `number_of_transformer_layers` is positive),
        the default `transformer_size` which is `None` breaks things. Thus,
        we need to set a reasonable default value so that the model works fine.
        """
        transformer_layers = config[NUM_TRANSFORMER_LAYERS]
        if isinstance(transformer_layers, dict):
            transformer_layers = transformer_layers[attribute]
        transformer_units = config[TRANSFORMER_SIZE]
        if isinstance(transformer_units, dict):
            transformer_units = transformer_units[attribute]
        if transformer_layers > 0 and (not transformer_units or transformer_units < 1):
            transformer_units = DEFAULT_TRANSFORMER_SIZE

        return transformer_layers, transformer_units

    def _prepare_transformer(
        self, attribute: Text, config: Dict[Text, Any]
    ) -> Tuple[int, int]:
        """Creates a transformer & returns its number of layers and output units."""
        transformer_layers, transformer_units = self._get_transformer_dimensions(
            attribute, config
        )
        self._tf_layers[self.TRANSFORMER] = prepare_transformer_layer(
            attribute_name=attribute,
            config=config,
            num_layers=transformer_layers,
            units=transformer_units,
            drop_rate=config[DROP_RATE],
            unidirectional=config[UNIDIRECTIONAL_ENCODER],
        )
        return transformer_layers, transformer_units

    def _prepare_masked_language_modeling(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        """Prepares masking and computing helper variables for masked language modeling.

        Only done for the text attribute and only if sequence-level (token-level)
        features are present (MLM requires token-level information).
        """
        if attribute == TEXT and SEQUENCE in attribute_signature and config[MASKED_LM]:
            self._enables_mlm = True
            self._tf_layers[self.MLM_INPUT_MASK] = layers.InputMask()

            # Unique IDs of different token types are needed to construct the possible
            # label space for MLM. If dense features are present, they're used as such
            # IDs, othwerise sparse features are embedded by a non-trainable
            # DenseForSparse layer to create small embeddings that serve as IDs.
            expect_dense_seq_features = any(
                [not signature.is_sparse for signature in attribute_signature[SEQUENCE]]
            )
            if not expect_dense_seq_features:
                self._tf_layers[
                    self.SPARSE_TO_DENSE_FOR_TOKEN_IDS
                ] = layers.DenseForSparse(
                    units=2,
                    use_bias=False,
                    trainable=False,
                    name=f"{self.SPARSE_TO_DENSE_FOR_TOKEN_IDS}.{attribute}",
                )

    def _calculate_output_units(
        self,
        attribute: Text,
        transformer_layers: int,
        transformer_units: int,
        config: Dict[Text, Any],
    ) -> int:
        """Determines the output units based on what layer components are present.

        The size depends on which component is the last created one in the internal
        pipeline that is `RasaFeatureCombiningLayer` -> `Ffnn` -> `Transformer`, since
        not all the components are always created.
        """
        # transformer is the last component
        if transformer_layers > 0:
            return transformer_units

        # the Ffnn block is the last component
        if len(config[HIDDEN_LAYERS_SIZES][attribute]) > 0:
            # this is the output size of the last layer of the Ffnn block
            return config[HIDDEN_LAYERS_SIZES][attribute][-1]

        # only the RasaFeatureCombiningLayer is present
        return self._tf_layers[self.FEATURE_COMBINING].output_units

    def _features_as_token_ids(
        self, features: List[Union[tf.Tensor, tf.SparseTensor]]
    ) -> Optional[tf.Tensor]:
        """Creates dense labels (token IDs) used for negative sampling in MLM."""
        # If there are dense features, we use them as labels - taking the first dense
        # feature in the list, but any other dense feature would do the job.
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                return tf.stop_gradient(f)

        # If no dense features are found, use a sparse feature but convert it into
        # a dense one first.
        for f in features:
            if isinstance(f, tf.SparseTensor):
                return tf.stop_gradient(
                    self._tf_layers[self.SPARSE_TO_DENSE_FOR_TOKEN_IDS](f)
                )

        return None

    def _create_mlm_tensors(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        seq_sent_features: tf.Tensor,
        mask_sequence: tf.Tensor,
        sentence_features_present: bool,
        training: bool,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Produces helper variables for masked language modelling (only in training).

        The `token_ids` embeddings can be viewed as token-level labels/unique IDs of all
        input tokens (to be used later in the MLM loss) because these embeddings aren't
        affected by dropout or masking and are effectively always unique for different
        input tokens (and same for the same tokens).
        `token_ids` share the batch and sequence dimension with the combined sequence-
        and sentence-level features, the last dimension is unimportant and mimics the
        first dense sequence-level feature in the list of features, or alternatively the
        last dimension will have size 2 if there are only sparse sequence features
        present.
        """
        token_ids = self._features_as_token_ids(sequence_features)

        # Pad in the sequence dimension to match the shape of combined sequence- and
        # sentence-level features. This means padding by 1 if sentence-level features
        # are present (those effectively have sequence length of 1) and not padding
        # otherwise.
        if sentence_features_present:
            token_ids = tf.pad(token_ids, [[0, 0], [0, 1], [0, 0]])
            mask_sequence = tf.pad(mask_sequence, [[0, 0], [0, 1], [0, 0]])

        # mlm_boolean_mask has the same shape as the tensor with all combined features
        # (except the last dimension), with True meaning tokens that are masked and
        # False meaning tokens that aren't masked or that are fake (padded) tokens.
        # Note that only sequence-level features are masked, nothing happens to the
        # sentence-level features in the combined features tensor.
        seq_sent_features, mlm_boolean_mask = self._tf_layers[self.MLM_INPUT_MASK](
            seq_sent_features, mask_sequence, training
        )

        return seq_sent_features, token_ids, mlm_boolean_mask

    def call(
        self,
        inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
        ],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Combines all of an attribute's features and embeds using a transformer.

        Arguments:
            inputs: Tuple containing:
                sequence_features: Dense or sparse tensors representing different
                    token-level features.
                sentence_features: Dense or sparse tensors representing different
                    sentence-level features.
                sequence_feature_lengths: A tensor containing the real sequence length
                    (the number of real -- not padding -- tokens) for each example in
                    the batch.
            training: A flag indicating whether the layer should behave in training mode
                (applying dropout to sparse tensors if applicable) or in inference mode
                (not applying dropout).

        Returns:
            outputs: Tensor with all features combined, masked (if doing MLM) and
                embedded with a transformer.
            seq_sent_features: Tensor with all features combined from just before the
                masking and transformer is applied
            mask_combined_sequence_sentence: A binary mask with 1s in place of real
                features in the combined feature tensor, and 0s in padded positions with
                fake features.
            token_ids: Tensor with dense token-level features which can serve as
                IDs (unique embeddings) of all the different tokens found in the batch.
                Empty tensor if not doing MLM.
            mlm_boolean_mask: A boolean mask with `True` where real tokens in `outputs`
                were masked and `False` elsewhere. Empty tensor if not doing MLM.
            attention_weights: Tensor containing self-attention weights received
                from the underlying transformer. Empty tensor if the transformer has 0
                layers.
        """
        sequence_features = inputs[0]
        sentence_features = inputs[1]
        sequence_feature_lengths = inputs[2]

        # Combine all features (sparse/dense, sequence-/sentence-level) into one tensor,
        # also get a binary mask that has 1s at positions with real features and 0s at
        # padded positions.
        seq_sent_features, mask_combined_sequence_sentence = self._tf_layers[
            self.FEATURE_COMBINING
        ]((sequence_features, sentence_features, sequence_feature_lengths))

        # Apply one or more dense layers.
        seq_sent_features = self._tf_layers[self.FFNN](seq_sent_features, training)

        # If using masked language modeling, mask the transformer inputs and get labels
        # for the masked tokens and a boolean mask. Note that TED does not use MLM loss,
        # hence using masked language modeling (if enabled) becomes just input dropout.
        if self._enables_mlm and training:
            mask_sequence = compute_mask(sequence_feature_lengths)
            (
                seq_sent_features_masked,
                token_ids,
                mlm_boolean_mask,
            ) = self._create_mlm_tensors(
                sequence_features,
                seq_sent_features,
                mask_sequence,
                sentence_features_present=len(sentence_features) > 0,
                training=training,
            )
        else:
            # tf.zeros((0,)) is an alternative to None
            token_ids = tf.zeros((0,))
            mlm_boolean_mask = tf.zeros((0,))
            seq_sent_features_masked = seq_sent_features

        # Apply the transformer (if present), hence reducing a sequences of features per
        # input example into a simple fixed-size embedding.
        if self._has_transformer:
            mask_padding = 1 - mask_combined_sequence_sentence
            outputs, attention_weights = self._tf_layers[self.TRANSFORMER](
                seq_sent_features_masked, mask_padding, training
            )
            outputs = tf.nn.gelu(outputs)
        else:
            # tf.zeros((0,)) is an alternative to None
            outputs, attention_weights = seq_sent_features_masked, tf.zeros((0,))

        return (
            outputs,
            seq_sent_features,
            mask_combined_sequence_sentence,
            token_ids,
            mlm_boolean_mask,
            attention_weights,
        )


def compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
    """Computes binary mask given real sequence lengths.

    Takes a 1-D tensor of shape `(batch_size,)` containing the lengths of sequences
    (in terms of number of tokens) in the batch. Creates a binary mask of shape
    `(batch_size, max_seq_length, 1)` with 1s at positions with real tokens and 0s
    elsewhere.
    """
    mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
    return tf.expand_dims(mask, -1)


def prepare_transformer_layer(
    attribute_name: Text,
    config: Dict[Text, Any],
    num_layers: int,
    units: int,
    drop_rate: float,
    unidirectional: bool,
) -> Union[
    TransformerEncoder,
    Callable[
        [tf.Tensor, Optional[tf.Tensor], Optional[Union[tf.Tensor, bool]]],
        Tuple[tf.Tensor, Optional[tf.Tensor]],
    ],
]:
    """Creates & returns a transformer encoder, potentially with 0 layers."""
    if num_layers > 0:
        return TransformerEncoder(
            num_layers,
            units,
            config[NUM_HEADS],
            units * 4,
            config[REGULARIZATION_CONSTANT],
            dropout_rate=drop_rate,
            attention_dropout_rate=config[DROP_RATE_ATTENTION],
            density=config[CONNECTION_DENSITY],
            unidirectional=unidirectional,
            use_key_relative_position=config[KEY_RELATIVE_ATTENTION],
            use_value_relative_position=config[VALUE_RELATIVE_ATTENTION],
            max_relative_position=config[MAX_RELATIVE_POSITION],
            name=f"{attribute_name}_encoder",
        )
    # create lambda so that it can be used later without the check
    return lambda x, mask, training: (x, None)
