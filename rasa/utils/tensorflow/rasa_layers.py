import tensorflow as tf
import tensorflow_addons as tfa
from typing import Text, List, Dict, Any, Union, Optional, Tuple

from rasa.core.constants import DIALOGUE
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import TEXT
from rasa.utils.tensorflow.model_data import FeatureSignature
from rasa.utils.tensorflow.constants import (
    REGULARIZATION_CONSTANT,
    WEIGHT_SPARSITY,
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
from rasa.utils.tensorflow.transformer import TransformerEncoder

# TODO: use this? it's in layers.py
tfa.options.TF_ADDONS_PY_OPS = True


class ConcatenateSparseDenseFeatures(tf.keras.layers.Layer):
    """Combines multiple sparse and dense feature tensors into one dense tensor.

    This layer combines features from various featurisers into a single feature array
    per input example. All features must be of the same feature type, i.e. sentence-
    level or sequence-level (token-level).

    A given list of tensors (whether sparse or dense) is turned into one tensor by:
    1. converting sparse tensors into dense ones
    2. optionally, applying dropout to sparse tensors before and/or after the conversion
    3. concatenating all tensors along the last dimension

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        feature_type: Feature type to be processed -- `sequence` or `sentence`.
        feature_type_signature: A list of `FeatureSignature`s for the given attribute
            and feature type.
        config: A model config for correctly parametrising the layer.

    Input shape:
        List of N-D tensors, each with shape: `(batch_size, ..., input_dim)`.
        All tensors must have the same shape, except the last dimension.
        All sparse tensors must have the same shape including the last dimension.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)` where `units` is the sum of
        the last dimension sizes across all input sensors, with sparse tensors instead
        contributing `config[DENSE_DIMENSION][attribute]` units each.

    Raises:
        RasaException if no feature signatures are provided.
    """

    def __init__(
        self,
        attribute: Text,
        feature_type: Text,
        feature_type_signature: List[FeatureSignature],
        config: Dict[Text, Any],
    ) -> None:
        if not feature_type_signature:
            raise RasaException(
                "The feature type signature must contain some feature signatures."
            )

        super().__init__(
            name=f"concatenate_sparse_dense_features_{attribute}_{feature_type}"
        )

        self.output_units = self._calculate_output_units(
            attribute, feature_type_signature, config
        )

        # Prepare dropout and sparse-to-dense layers if any sparse tensors are expected
        self._tf_layers = {}
        if any([signature.is_sparse for signature in feature_type_signature]):
            self._prepare_layers_for_sparse_tensors(attribute, feature_type, config)

    def _prepare_layers_for_sparse_tensors(
        self, attribute: Text, feature_type: Text, config: Dict[Text, Any],
    ) -> None:
        """Sets up sparse tensor pre-processing before combining with dense ones."""
        # For optionally applying dropout to sparse tensors
        if config[SPARSE_INPUT_DROPOUT]:
            self._tf_layers["sparse_dropout"] = layers.SparseDropout(
                rate=config[DROP_RATE]
            )

        # For converting sparse tensors to dense
        self._tf_layers["sparse_to_dense"] = layers.DenseForSparse(
            name=f"sparse_to_dense.{attribute}_{feature_type}",
            units=config[DENSE_DIMENSION][attribute],
            reg_lambda=config[REGULARIZATION_CONSTANT],
        )

        # For optionally apply dropout to sparse tensors after they're converted to
        # dense tensors.
        if config[DENSE_INPUT_DROPOUT]:
            self._tf_layers["dense_dropout"] = tf.keras.layers.Dropout(
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
        if "sparse_dropout" in self._tf_layers:
            feature = self._tf_layers["sparse_dropout"](feature, training)

        feature = self._tf_layers["sparse_to_dense"](feature)

        if "dense_dropout" in self._tf_layers:
            feature = self._tf_layers["dense_dropout"](feature, training)

        return feature

    def call(
        self,
        inputs: Tuple[List[Union[tf.Tensor, tf.SparseTensor]]],
        training: bool = False,
    ) -> tf.Tensor:
        """Combines sparse and dense feature tensors into one tensor.

        Arguments:
            inputs: Tuple containing one list of tensors, all of the same rank.
            training: Python boolean indicating whether the layer should behave in
                training mode (applying dropout to sparse tensors if applicable) or in
                inference mode (not applying dropout).

        Returns:
            Single tensor of the same shape as the input tensors, except the last
            dimension.
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


class RasaFeatureCombiningLayer(tf.keras.layers.Layer):
    """Combines multiple dense or sparse feature tensors into one.

    This layer combines features by following these steps:
    1. Apply a `ConcatenateSparseDenseFeatures` layer separately to sequence- and
        sentence-level features, yielding two tensors (one for each feature type).
    2. Concatenate the sequence- and sentence-level features into one tensor.

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        attribute_signature: A dictionary containing two lists of `FeatureSignature`s,
            one for each feature type of the given attribute.
        config: A model config used for correctly parametrising the the layer and the
            `ConcatenateSparseDenseFeatures` layer it creates internally.

    Input shape:
        Tuple of four tensor inputs:
            sequence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
                different for sparse vs dense tensors.
            sentence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, 1, input_dim)` where `input_dim` can be different for
                sparse vs dense tensors, and can differ from that in `sequence_features`.
            sequence_feature_lengths: Dense tensor of shape `(batch_size, )` containing
                the real sequence length for each example in the batch, i.e. the lengths
                of the real (not padded) sequence-level (token-level) features.

    Output shape:
        combined features: a 3-D tensor with shape `(batch_size, sequence_length, units)`
            where `units` is  completely  determined by the internally applied
            `ConcatenateSparseDenseFeatures` layer and `sequence_length` is the combined
            length of sequence- and sentence-level features (`max_seq_length + 1` if
            both sequence- and sentence-level features are present, `max_seq_length` if
            only sequence-level features are present, and 1 if only sentence-level
            features are present).
        mask_combined_sequence_sentence: a 3-D tensor with shape
            `(batch_size, sequence_length, 1)`.

    Raises:
        A RasaException if no feature signatures are provided.
    """

    def __init__(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        if not attribute_signature or not (
            attribute_signature.get(SENTENCE, [])
            or attribute_signature.get(SEQUENCE, [])
        ):
            raise RasaException(
                "The attribute signature must contain some feature signatures."
            )

        super().__init__(name=f"rasa_feature_combining_layer_{attribute}")

        self._tf_layers = {}

        # Prepare sparse-dense combining layers for each present feature type
        self._feature_types_present = self._get_present_feature_types(
            attribute_signature
        )
        self._prepare_sparse_dense_concat_layers(attribute, attribute_signature, config)

        # Prepare components for combining sequence- and sentence-level features
        self.output_units = self._prepare_sequence_sentence_concat(attribute, config)

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
                and attribute_signature[feature_type]
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
    ) -> int:
        """Sets up combining sentence- and sequence-level features if needed.

        Returns the number of output units for this layer class.
        """
        if (
            self._feature_types_present[SEQUENCE]
            and self._feature_types_present[SENTENCE]
        ):
            # The output units of this layer will be based on the output sizes of the
            # sparse+dense combining layers that are internally applied to all features.
            sequence_units = self._tf_layers[f"sparse_dense.{SEQUENCE}"].output_units
            sentence_units = self._tf_layers[f"sparse_dense.{SENTENCE}"].output_units

            # Last dimension needs to be unified if sequence- and sentence-level features
            # have different sizes, e.g. due to being produced by different featurizers.
            if sequence_units != sentence_units:
                for feature_type in [SEQUENCE, SENTENCE]:
                    self._tf_layers[
                        f"unify_dims_before_seq_sent_concat.{feature_type}"
                    ] = layers.Ffnn(
                        layer_name_suffix=f"unify_dims.{attribute}_{feature_type}",
                        layer_sizes=[config[CONCAT_DIMENSION][attribute]],
                        dropout_rate=config[DROP_RATE],
                        reg_lambda=config[REGULARIZATION_CONSTANT],
                        sparsity=config[WEIGHT_SPARSITY],
                    )
                return config[CONCAT_DIMENSION][attribute]
            else:
                # If the features have the same last dimension size, that will also be
                # the output size of the entire layer.
                return sequence_units

        # If only sequence-level features are present, they will determine the output
        # size of this layer.
        elif self._feature_types_present[SEQUENCE]:
            return self._tf_layers[f"sparse_dense.{SEQUENCE}"].output_units

        # If only sentence-level features are present, they will determine the output
        # size of this layer.
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
        # summing the two padded feature arrays like this (batch size  = 1):
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
                sequence_features: List of 3-D dense or sparse tensors with token-level
                    features.
                sentence_features: List of 3-D dense or sparse tensors with sentence-
                    level features.
                sequence_feature_lengths: Dense 1-D tensor containing the lengths of
                    sequence-level features for the batch.
            training: Python boolean indicating whether the layer should behave in
                training mode (applying dropout to sparse tensors if applicable) or in
                inference mode (not applying dropout).

        Returns:
            combined features: a 3-D tensor containing the combined sequence- and
                sentence-level, sparse & dense features.
            mask_combined_sequence_sentence: a binary 3-D tensor with 1s in place of
                real features in the combined feature array, and 0s in place of fake
                features.
        """
        sequence_features = inputs[0]
        sentence_features = inputs[1]
        sequence_feature_lengths = inputs[2]

        # This mask is specifically for sequence-level features.
        mask_sequence = _compute_mask(sequence_feature_lengths)

        sequence_features_combined = self._combine_sequence_level_features(
            sequence_features, mask_sequence, training
        )

        (
            sentence_features_combined,
            combined_sequence_sentence_feature_lengths,
        ) = self._combine_sentence_level_features(
            sentence_features, sequence_feature_lengths, training
        )

        mask_combined_sequence_sentence = _compute_mask(
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


class RasaSequenceLayer(tf.keras.layers.Layer):
    """Creates an embedding from all features for a sequence attribute; facilitates MLM.

    This layer combines all features for an attribute and embeds them using a
    transformer. The layer is meant only for attributes with sequence-level features,
    such as `text`, `response` and `action_text`.

    Internally, this layer uses `RasaFeatureCombiningLayer` and applies the following
    steps:
    1. Combine features using `RasaFeatureCombiningLayer`.
    2. Apply a dense layer(s) to the combined features.
    3. Optionally (during training for the `text` attribute), apply masking to the
        features and create further helper variables for masked language modeling.
    4. Embed the features using a transformer, effectively going from variable-length
        sequences of features to fixed-size input example embeddings.

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        attribute_signature: A dictionary containing two lists of `FeatureSignature`s,
            one for each feature type of the given attribute.
        config: A model config used for correctly parameterising the underlying layers.

    Input shape:
        Tuple of four tensor inputs:
            sequence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
                different for sparse vs dense tensors.
            sentence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, 1, input_dim)` where `input_dim` can be different for
                sparse vs dense tensors, and can differ from that in `sequence_features`.
            sequence_feature_lengths: Dense tensor of shape `(batch_size, )` containing
                the real sequence lengths for the sequence-level features, i.e. the
                numbers of real (not padding) tokens for each example in the batch.

    Output shape:
        outputs: `(batch_size, seq_length, units)` where `units` matches the underlying
            transformer's output size if the transformer has some layers, otherwise
            `units` matches that of the `Ffnn` block applied to the combined features, or
            it's the output size of the underlying `RasaFeatureCombiningLayer` if the
            `Ffnn` block has 0 layers. `seq_length` is the sum of the sequence dimension
            sizes of sequence- and sentence-level features (for details, see the output
            shape of `RasaFeatureCombiningLayer`). If both feature types are present,
            then `seq_length` will be 1 + the length of the longest sequence of real
            tokens across all examples in the given batch.
        seq_sent_features: `(batch_size, seq_length, hidden_dim)`, where `hidden_dim` is
            the output size of the underlying `Ffnn` block, or the output size of the
            underlying `RasaFeatureCombiningLayer` if the `Ffnn` block has 0 layers.
        mask_combined_sequence_sentence: `(batch_size, seq_length, hidden_dim)`
        token_ids: `(batch_size, seq_length, id_dim)` where `id_dim` is unimportant and
            it's the last-dimension size of the first sequence-level dense feature
            if any is present, and 2 otherwise. Empty tensor if not doing MLM.
        mlm_boolean_mask: `(batch_size, seq_length, 1)`, empty tensor if not doing MLM.
        attention_weights: `(transformer_layers, batch_size, num_transformer_heads,
            seq_length, seq_length)`, empty tensor if the transformer has 0 layers.

    Raises:
        A `RasaException` if no feature signatures for sequence-level features are
            provided.
    """

    def __init__(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        if not attribute_signature or not attribute_signature.get(SEQUENCE, []):
            raise RasaException(
                "The attribute signature must contain some sequence-level feature\
                signatures but none were found."
            )

        super().__init__(name=f"rasa_sequence_layer_{attribute}")

        self._tf_layers: Dict[Text, Any] = {
            "feature_combining_layer": RasaFeatureCombiningLayer(
                attribute, attribute_signature, config
            ),
            "ffnn": layers.Ffnn(
                config[HIDDEN_LAYERS_SIZES][attribute],
                config[DROP_RATE],
                config[REGULARIZATION_CONSTANT],
                config[WEIGHT_SPARSITY],
                layer_name_suffix=attribute,
            ),
        }

        self._prepare_masked_language_modeling(attribute, attribute_signature, config)

        transformer_layers, transformer_units = self._get_transformer_dimensions(
            attribute, config
        )
        self._prepare_transformer(
            attribute, config, transformer_layers, transformer_units
        )

        self.output_units = self._calculate_output_units(
            attribute, transformer_layers, transformer_units, config
        )

    def _prepare_transformer(
        self,
        attribute: Text,
        config: Dict[Text, Any],
        transformer_layers: int,
        transformer_units: int,
    ) -> None:
        """Prepares a transformer if the config specifies >0 transformer layers."""
        if transformer_layers > 0:
            self._tf_layers["transformer"] = TransformerEncoder(
                num_layers=transformer_layers,
                units=transformer_units,
                num_heads=config[NUM_HEADS],
                filter_units=transformer_units * 4,
                reg_lambda=config[REGULARIZATION_CONSTANT],
                dropout_rate=config[DROP_RATE],
                attention_dropout_rate=config[DROP_RATE_ATTENTION],
                sparsity=config[WEIGHT_SPARSITY],
                unidirectional=config[UNIDIRECTIONAL_ENCODER],
                use_key_relative_position=config[KEY_RELATIVE_ATTENTION],
                use_value_relative_position=config[VALUE_RELATIVE_ATTENTION],
                max_relative_position=config[MAX_RELATIVE_POSITION],
                name=f"{attribute}_encoder",
            )

    @staticmethod
    def _get_transformer_dimensions(
        attribute: Text, config: Dict[Text, Any]
    ) -> Tuple[int, int]:
        """Determines # of transformer layers & output size from the model config.

        The config can contain these directly (same for all attributes) or specified
        separately for each attribute.
        """
        transformer_layers = config[NUM_TRANSFORMER_LAYERS]
        if isinstance(transformer_layers, dict):
            transformer_layers = transformer_layers[attribute]
        transformer_units = config[TRANSFORMER_SIZE]
        if isinstance(transformer_units, dict):
            transformer_units = transformer_units[attribute]

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
            self._tf_layers["mlm_input_mask"] = layers.InputMask()

            # Unique IDs of different token types are needed to construct the possible
            # label space for MLM. If dense features are present, they're used as such
            # IDs, othwerise sparse features are embedded by a non-trainable
            # DenseForSparse layer to create small embeddings that serve as IDs.
            expect_dense_seq_features = any(
                [not signature.is_sparse for signature in attribute_signature[SEQUENCE]]
            )
            if not expect_dense_seq_features:
                self._tf_layers["sparse_to_dense_token_ids"] = layers.DenseForSparse(
                    units=2,
                    use_bias=False,
                    trainable=False,
                    name=f"sparse_to_dense_token_ids.{attribute}",
                )
        else:
            self._enables_mlm = False

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
        return self._tf_layers["feature_combining_layer"].output_units

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
                return tf.stop_gradient(self._tf_layers["sparse_to_dense_token_ids"](f))

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
        seq_sent_features, mlm_boolean_mask = self._tf_layers["mlm_input_mask"](
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
        """Combines all attribute features into one and embed using a transformer.

        Arguments:
            inputs: Tuple containing:
                sequence_features: List of 3-D dense or sparse tensors with token-level
                    features.
                sentence_features: List of 3-D dense or sparse tensors with sentence-
                    level features.
                mask_sequence: a 3-D tensor mask that has 1s at real and 0s at padded
                    positions corresponding to tokens in `sequence_features`.
                mask_combined_sequence_sentence: a 3-D tensor mask similar to
                    `mask_sequence` but having each sequence of 1s longer by 1 to
                    account for sequence lengths of sequence- and sentence-level
                    features being combined.
            training: Python boolean indicating whether the layer should behave in
                training mode (applying dropout to sparse tensors if applicable) or in
                inference mode (not applying dropout).

        Returns:
            outputs: 3-D tensor with all features combined, optionally masked (when
                doing MLM) and embedded by a transformer.
            seq_sent_features: 3-D tensor, like `outputs`, but without masking and
                transformer applied.
            token_ids: 3-D tensor with dense token-level features which can serve as
                unique embeddings/IDs of all the different tokens found in the batch.
                Empty tensor if not doing MLM.
            mlm_boolean_mask: 3-D tensor mask that has 1s where real tokens in `outputs`
                were masked and 0s elsewhere. Empty tensor if not doing MLM.
            attention_weights: 5-D tensor containing self-attention weights received
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
            "feature_combining_layer"
        ]((sequence_features, sentence_features, sequence_feature_lengths))

        # Apply one or more dense layers.
        seq_sent_features = self._tf_layers["ffnn"](seq_sent_features, training)

        # If using masked language modeling, mask the transformer inputs and get labels
        # for the masked tokens and a boolean mask.
        if self._enables_mlm and training:
            mask_sequence = _compute_mask(sequence_feature_lengths)
            seq_sent_features, token_ids, mlm_boolean_mask = self._create_mlm_tensors(
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

        # Apply the transformer (if present), hence reducing a sequences of features per
        # input example into a simple fixed-size embedding.
        if "transformer" in self._tf_layers:
            mask_padding = 1 - mask_combined_sequence_sentence
            outputs, attention_weights = self._tf_layers["transformer"](
                seq_sent_features, mask_padding, training
            )
            outputs = tfa.activations.gelu(outputs)
        else:
            # tf.zeros((0,)) is an alternative to None
            outputs, attention_weights = seq_sent_features, tf.zeros((0,))

        return (
            outputs,
            seq_sent_features,
            mask_combined_sequence_sentence,
            token_ids,
            mlm_boolean_mask,
            attention_weights,
        )


def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
    """Computes binary mask given real sequence lengths.

    Takes a 1-D tensor of shape `(batch_size,)` containing the lengths of real feature
    sequences in the batch. Creates a binary mask of shape `(batch_size, max_seq_length,
    1)` with 1s at positions with real features and 0s elsewhere.
    """
    mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
    return tf.expand_dims(mask, -1)
