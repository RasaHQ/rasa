import tensorflow as tf
from typing import Text, List, Dict, Any, Union, Optional, Tuple
import tensorflow_addons as tfa

from rasa.core.constants import DIALOGUE
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
from rasa.utils.tensorflow.models import TransformerRasaModel
from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.utils.tensorflow.model_data_utils import _compute_mask


# TODO: use this? it's in layers.py
tfa.options.TF_ADDONS_PY_OPS = True

# TODO: check for empty lists of features where necessary
# TODO: add return types to helper functions


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
        attribute_signature: A list of `FeatureSignature`s for the given attribute and
            feature type.
        config: A model config for correctly parametrising the layer.

    Input shape:
        List of N-D tensors, each with shape: `(batch_size, ..., input_dim)`.
        All tensors must have the same shape, except the last dimension.
        All sparse tensors must have the same shape including the last dimension.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)` where `units` is the sum of
        the last dimension sizes across all input sensors, with sparse tensors instead
        contributing `config[DENSE_DIMENSION][attribute]` units each.
    """

    def __init__(
        self,
        attribute: Text,
        feature_type: Text,
        attribute_signature: List[FeatureSignature],
        config: Dict[Text, Any],
    ) -> None:
        super().__init__(
            name=f"concatenate_sparse_dense_features_{attribute}_{feature_type}"
        )

        self.output_units = self._calculate_output_units(
            attribute, attribute_signature, config
        )

        # Prepare dropout and sparse-to-dense layers if any sparse tensors are expected
        self._tf_layers = {}
        if any([signature.is_sparse for signature in attribute_signature]):
            self._prepare_layers_for_sparse_tensors(attribute, feature_type, config)

    def _prepare_layers_for_sparse_tensors(
        self, attribute: Text, feature_type: Text, config: Dict[Text, Any],
    ):
        # For optionally apply dropout to sparse tensors
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

    def _calculate_output_units(
        self,
        attribute: Text,
        attribute_signature: List[FeatureSignature],
        config: Dict[Text, Any],
    ):
        """Determine the output units from the provided feature signatures.

        Sparse features will be turned into dense ones, hence they each contribute with
        their future dense number of units.
        """
        return sum(
            [
                config[DENSE_DIMENSION][attribute]
                if signature.is_sparse
                else signature.units
                for signature in attribute_signature
            ]
        )

    def _process_sparse_feature(self, feature: tf.SparseTensor, training: bool):
        """Turn sparse tensor into dense, maybe apply dropout before and/or after."""
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
        """Combine sparse and dense feature tensors into one tensor.

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

        return tf.concat(dense_features, axis=-1)


class ConcatenateSequenceSentenceFeatures(tf.keras.layers.Layer):
    """Combines two tensors (sentence-level and sequence level features) into one.

    This layer concatenates sentence- and sequence-level (token-level) features (each
    feature type represented by a single dense tensor) into a single feature tensor.
    When expanded to the same shape, sentence-level features can be viewed as sequence-
    level ones, but with the sequence length being 1 for all examples. Hence, features
    of the two types can be concatenated along the sequence (token) dimension,
    effectively increasing the sequence length of each example by 1. Because sequence-
    level features are all padded to the same max. sequence length, the concatenation is
    done in the following steps:
    1. Optional: If the last dimension size of sequence- and sentence-level feature
        differs, it's unified by applying a dense layer to each feature types.
    1. sentence-level features are masked out everywhere except at the first available
        position (`sequence_length+1`, where the length varies across input examples)
    2. the padding of sequence-level features is increased by 1 to prepare empty space
        for sentence-level features
    3. sentence-level features are added just after the sequence-level ones

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        sequence_units: Last dimension size of the expected sequence-level features.
        sentence_units: Last dimension size of the expected sentence-level features.
        config: A model config for correctly parametrising the layer's components.

    Input shape:
        Tuple of three 3-D dense tensors, with shapes:
            sequence: `(batch_size, max_seq_length, input_dim_seq)`
            sentence: `(batch_size, 1, input_dim_sent)`
            mask: `(batch_size, max_seq_length+1, 1)`

    Output shape:
        3-D tensor with shape: `(batch_size, sequence_length, units)` where:
        - `units` matches `input_dim_seq` if that is the same as `input_dim_sent`,
            otherwise `units = config[CONCAT_DIMENSION][attribute]`.
        - `sequence_length` is the sum of 2nd dimension sizes of arguments `sequence`
            and`sentence`.
    """

    def __init__(
        self,
        attribute: Text,
        sequence_units: Optional[int],
        sentence_units: Optional[int],
        config: Dict[Text, Any],
    ) -> None:
        super().__init__(name=f"concatenate_sequence_sentence_features_{attribute}")

        self._tf_layers = {}

        # main use case -- both sequence and sentence features are expected
        if sequence_units and sentence_units:
            self._return_feature_type = f"{SEQUENCE}_{SENTENCE}"

            # prepare dimension unifying layers if needed
            if sequence_units != sentence_units:
                self.output_units = config[CONCAT_DIMENSION][attribute]
                self._prepare_dimension_unifying_layers(attribute, config)
            else:
                self.output_units = sequence_units

        # edge cases where only sequence or only sentence features are expected
        elif sequence_units and not sentence_units:
            self._return_feature_type = SEQUENCE
            self.output_units = sequence_units
        else:
            self._return_feature_type = SENTENCE
            self.output_units = sentence_units

    def _prepare_dimension_unifying_layers(
        self, attribute: Text, config: Dict[Text, Any]
    ):
        for feature_type in [SEQUENCE, SENTENCE]:
            self._tf_layers[f"unify_dims.{feature_type}"] = layers.Ffnn(
                layer_name_suffix=f"unify_dims.{attribute}_{feature_type}",
                layer_sizes=[config[CONCAT_DIMENSION][attribute]],
                dropout_rate=config[DROP_RATE],
                reg_lambda=config[REGULARIZATION_CONSTANT],
                sparsity=config[WEIGHT_SPARSITY],
            )

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Concatenate sequence- and sentence-level feature tensors into one tensor.

        Arguments:
            inputs: Tuple containing three dense 3-D tensors.

        Returns:
            Single dense 3-D tensor containing the concatenated sequence- and sentence-
            level features.
        """
        sequence = inputs[0]
        sentence = inputs[1]
        mask = inputs[2]

        # Combine both feature types when they're present
        if self._return_feature_type == f"{SEQUENCE}_{SENTENCE}":
            if f"unify_dims.{SEQUENCE}" in self._tf_layers:
                sequence = self._tf_layers[f"unify_dims.{SEQUENCE}"](sequence)
            if f"unify_dims.{SENTENCE}" in self._tf_layers:
                sentence = self._tf_layers[f"unify_dims.{SENTENCE}"](sentence)

            # mask has for each input example a sequence of 1s of length seq_length+1,
            # where seq_length is the number of real tokens. The rest is 0s which form
            # a padding up to the max. sequence length + 1 (max. # of real tokens + 1).
            # Here the mask is turned into a mask that has 0s everywhere and 1 only at
            # the immediate next position after the last real token's position for given
            # input example. Example (batch size 2, sequence lengths [1, 2]):
            # [[[1], [0], [0]],     ___\   [[[0], [1], [0]],
            #  [[1], [1], [0]]]        /    [[0], [0], [1]]]
            last = mask * tf.math.cumprod(
                1 - mask, axis=1, exclusive=True, reverse=True
            )

            # The new mask is used to distribute the sentence features at the sequence
            # positions marked by 1s. The sentence features' dimensionality effectively
            # changes from `(batch_size, 1, feature_dim)` to `(batch_size, max_seq_length+1,
            # feature_dim)`, but the array is sparse, with real features present only at
            # positions determined by 1s in the mask.
            sentence = last * sentence

            # Padding of sequence-level features is increased by 1 in the sequence
            # dimension to match the shape of modified sentence-level features.
            sequence = tf.pad(sequence, [[0, 0], [0, 1], [0, 0]])

            # Sequence- and sentence-level features effectively get concatenated by
            # summing the two padded feature arrays like this (batch size  = 1):
            # [[seq1, seq2, seq3, 0, 0]] + [[0, 0, 0, sent1, 0]] =
            # = [[seq1, seq2, seq3, sent1, 0]]
            return sequence + sentence

        # Return just the sequence-level features because no sentence-level features are
        # present.
        elif self._return_feature_type == SEQUENCE:
            return sequence

        # Return just the sentence-level features because no sequence-level features are
        # present.
        return sentence


class RasaFeatureCombiningLayer(tf.keras.layers.Layer):
    """Combines multiple dense or sparse feature tensors into one.

    This layer combines features by following these steps:
    1. Apply a `ConcatenateSparseDenseFeatures` layer separately to sequence- and
        sentence-level features, yielding two tensors (one for each feature type).
    2. Apply a `ConcatenateSequenceSentenceFeatures` layer to the two tensors to combine
        sequence- and sentence-level features into one tensor.

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        attribute_signature: A dictionary containing two lists of `FeatureSignature`s, 
            one for each feature type of the given attribute.
        config: A model config used for correctly parametrising the `ConcatenateSparse
            DenseFeatures` and `ConcatenateSequenceSentenceFeatures` layers.

    Input shape:
        Tuple of four tensor inputs:
            sequence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
                different for sparse vs dense tensors.
            sentence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, 1, input_dim)` where `input_dim` can be different for
                sparse vs dense tensors, and can differ from that in `sequence_features`.
            mask_sequence: dense 3-D tensor with the shape `(batch_size, max_sequence_
                length, 1)`.
            mask_combined_sequence_sentence: dense 3-D tensor with the shape `(batch_size,
                max_sequence_length+1, 1)`, i.e. with the 2nd dimension combining the
                lengths of sequence- and sentence-level features.

    Output shape:
        3-D tensor with shape: `(batch_size, sequence_length, units)` where `units` is 
        completely  determined by the internally applied `ConcatenateSparseDenseFeatures`
        layer and `sequence_length` is completely determined by the internally applied 
        `ConcatenateSequenceSentenceFeatures` layer.

    Raises:
        A ValueError if no feature signatures are provided.
    """

    def __init__(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        if not attribute_signature or not (
            len(attribute_signature.get(SENTENCE, [])) > 0
            or len(attribute_signature.get(SEQUENCE, [])) > 0
        ):
            raise ValueError("The data signature must contain some features.")

        super().__init__(name=f"rasa_feature_combining_layer_{attribute}")

        self._tf_layers = {}

        # 1. prepare layers for combining sparse and dense features
        self._prepare_sparse_dense_concat_layers(attribute, attribute_signature, config)

        # 2. prepare layer for combining sequence- and sentence-level features
        sentence_units = (
            None
            if f"sparse_dense.{SENTENCE}" not in self._tf_layers
            else self._tf_layers[f"sparse_dense.{SENTENCE}"].output_units
        )
        sequence_units = (
            None
            if f"sparse_dense.{SEQUENCE}" not in self._tf_layers
            else self._tf_layers[f"sparse_dense.{SEQUENCE}"].output_units
        )
        self._tf_layers["concat_seq_sent"] = ConcatenateSequenceSentenceFeatures(
            attribute=attribute,
            sequence_units=sequence_units,
            sentence_units=sentence_units,
            config=config,
        )

        self.output_units = self._tf_layers["concat_seq_sent"].output_units

    def _prepare_sparse_dense_concat_layers(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ):
        for feature_type in attribute_signature:
            # Prepare the concatenation layer only if any features are expected for this
            # feature type.
            if attribute_signature[feature_type] and feature_type in [
                SENTENCE,
                SEQUENCE,
            ]:
                self._tf_layers[
                    f"sparse_dense.{feature_type}"
                ] = ConcatenateSparseDenseFeatures(
                    attribute=attribute,
                    feature_type=feature_type,
                    attribute_signature=attribute_signature[feature_type],
                    config=config,
                )

    def call(
        self,
        inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
        ],
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Combine multiple 3-D dense/sparse feature tensors into one.

        Arguments:
            inputs: Tuple containing:
                sequence_features: List of 3-D dense or sparse tensors with token-level
                    features.
                sentence_features: List of 3-D dense or sparse tensors with sentence-level
                    features.
                mask_sequence: a 3-D tensor mask that has 1s at real and 0s at padded
                    positions corresponding to tokens in `sequence_features`.
                mask_combined_sequence_sentence: a 3-D tensor mask similar to 
                    `mask_sequence` but having each sequence of 1s longer by 1 to account
                    for sequence lengths of sequence- and sentence-level features being
                    combined.
            training: Python boolean indicating whether the layer should behave in
                training mode (applying dropout to sparse tensors if applicable) or in
                inference mode (not applying dropout).

        Returns:
            Single dense 3-D tensor containing the combined sequence- and sentence-
            level, sparse & dense features.
        """
        sequence_features = inputs[0]
        sentence_features = inputs[1]
        sequence_feature_lengths = inputs[2]

        mask_sequence = _compute_mask(sequence_feature_lengths)
        combined_sequence_sentence_features_lengths = sequence_feature_lengths
        if len(sentence_features) > 0:
            combined_sequence_sentence_features_lengths += 1
        mask_combined_sequence_sentence = _compute_mask(
            combined_sequence_sentence_features_lengths
        )

        if f"sparse_dense.{SEQUENCE}" in self._tf_layers:
            sequence = self._tf_layers[f"sparse_dense.{SEQUENCE}"](
                (sequence_features,), training=training
            )

            # apply mask which has 1s at positions of real tokens and 0s at all padded
            # token positions. This is needed because the sparse+dense combining layer
            # might've turned some fake (padded) features (i.e. 0s) into non-zero numbers
            # and we want those to become zeros again.
            # This step isn't needed for sentence-level features because those are never
            # padded -- the effective sequence length in their case is always 1.
            sequence = sequence * mask_sequence
        else:
            sequence = None

        if f"sparse_dense.{SENTENCE}" in self._tf_layers:
            sentence = self._tf_layers[f"sparse_dense.{SENTENCE}"](
                (sentence_features,), training=training
            )
        else:
            sentence = None

        combined_sequence_sentence_features = self._tf_layers["concat_seq_sent"](
            (sequence, sentence, mask_combined_sequence_sentence)
        )

        return combined_sequence_sentence_features, mask_combined_sequence_sentence


class RasaSequenceLayer(tf.keras.layers.Layer):
    """Creates an embedding from all features for a sequence attribute; facilitates MLM.

    This layer combines all features for an attribute and embeds them using a 
    transformer. The layer is meant only for attributes with sequence-level features,
    such as `text` and `action_text`.

    Internally, this layer extends RasaFeatureCombiningLayer and goes through the following steps:
    1. Combine features using RasaFeatureCombiningLayer.
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
        config: A model config used for correctly parametrising the underlying layers.

    Input shape:
        Tuple of four tensor inputs:
            sequence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
                different for sparse vs dense tensors.
            sentence_features: List of 3-D dense or sparse tensors, each with shape
                `(batch_size, 1, input_dim)` where `input_dim` can be different for
                sparse vs dense tensors, and can differ from that in `sequence_features`.
            mask_sequence: dense 3-D tensor with the shape `(batch_size, max_sequence_
                length, 1)`.
            mask_combined_sequence_sentence: dense 3-D tensor with the shape `(batch_size,
                max_sequence_length+1, 1)`, i.e. with the 2nd dimension combining the 
                lengths of sequence- and sentence-level features.

    Output shape:
        outputs: `(batch_size, max_seq_length+1, units)` where `units` matches
            the underlying transformer's output size if the transformer has some layers,
            otherwise `units` matches that of the Ffnn block applied to the combined 
            features, or it's the output size of the underlying `RasaFeatureCombiningLayer` the 
            Ffnn block has 0 layers. `max_seq_length` is the length of the longest
            sequence of tokens in the given batch.
        seq_sent_features: `(batch_size, max_seq_length+1, hidden_dim)`, where 
            `hidden_dim` is the output size of the underlying Ffnn block, or the output
            size of the underlying `RasaFeatureCombiningLayer` if the Ffnn block has 0 layers.
        token_ids: `(batch_size, max_seq_length+1, id_dim)` where id_dim is unimportant
            and it's the last-dimension size of the first sequence-level dense feature
            if any is present, and 2 otherwise. Empty tensor if not doing MLM.
        mlm_boolean_mask: `(batch_size, max_seq_length+1, 1)`, empty tensor if not doing MLM.
        attention_weights: `(transformer_layers, batch_size, num_transformer_heads, 
            max_seq_length+1, max_seq_length+1)`, empty tensor if the transformer has 0
            layers.

    Raises:
        A ValueError if no feature signatures for sequence-level features are provided.
    """

    def __init__(
        self,
        attribute: Text,
        attribute_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        if not attribute_signature or len(attribute_signature.get(SEQUENCE, [])) == 0:
            raise ValueError(
                "The data signature must contain some sequence-level features but none\
                were found."
            )

        super().__init__(name=f"rasa_sequence_layer_{attribute}")

        self._tf_layers = {
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

        transformer_layers, transformer_units = self._prepare_transformer(
            attribute, config
        )

        self.output_units = self._calculate_output_units(
            attribute, transformer_layers, transformer_units, config
        )

    def _prepare_transformer(self, attribute: Text, config: Dict[Text, Any]):
        transformer_layers, transformer_units = self._get_transformer_dimensions(
            attribute, config
        )
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
        return transformer_layers, transformer_units

    def _get_transformer_dimensions(self, attribute: Text, config: Dict[Text, Any]):
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
    ):
        """Prepare masking and computing helper variables for masked language modeling.

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
    ):
        """Determine the output units based on which layer components are present.

        The output units depend on which component is the last one in the internal 
        pipeline that is `RasaFeatureCombiningLayer` -> `Ffnn` -> `Transformer`, because not all 
        the components are necessarily created.
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
        # feature in the list because any dense feature will do the job.
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
        mask_combined_sequence_sentence: tf.Tensor,
        training: bool,
    ):
        """Produce helper variables for masked language modelling (only in training).
        
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
        # sentence-level features (sentence-level features effectively have sequence
        # length of 1).
        token_ids = tf.pad(token_ids, [[0, 0], [0, 1], [0, 0]])

        # mlm_boolean_mask has the same shape as mask_combined_sequence_sentence (i.e. as
        # the tensor with all combined features), with True meaning tokens that are
        # masked and False meaning tokens that aren't masked or that are fake
        # (padded) tokens.
        seq_sent_features, mlm_boolean_mask = self._tf_layers["mlm_input_mask"](
            seq_sent_features, mask_combined_sequence_sentence, training
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
        """Combine all features for an attribute into one and embed using a transformer.

        Arguments:
            inputs: Tuple containing:
                sequence_features: List of 3-D dense or sparse tensors with token-level
                    features.
                sentence_features: List of 3-D dense or sparse tensors with sentence-level
                    features.
                mask_sequence: a 3-D tensor mask that has 1s at real and 0s at padded
                    positions corresponding to tokens in `sequence_features`.
                mask_combined_sequence_sentence: a 3-D tensor mask similar to 
                    `mask_sequence` but having each sequence of 1s longer by 1 to account
                    for sequence lengths of sequence- and sentence-level features being 
                    combined.
            training: Python boolean indicating whether the layer should behave in
                training mode (applying dropout to sparse tensors if applicable) or in
                inference mode (not applying dropout).

        Returns:
            outputs: 3-D tensor with all features combined, optionally masked (when doing
                MLM) and embedded by a transformer.
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

        # TODO
        # Get mask for sequence features only, with 1s for real tokens and 0 for padded.
        # This one will be None if the sequence lengths aren't found in tf_batch_data.
        # mask_sequence = self._get_mask_for(tf_batch_data, TEXT, SEQUENCE_LENGTH)

        # TODO
        # Create mask for sequence- and sentence-leve features combined. Differs from
        # the previous mask only in being longer by 1 in the sequence dimension, due to
        # having one more 1 at the end of each sequence (representing the sentence-level
        # featues).
        # This call can break if the sequence lengths aren't found in tf_batch_data,
        # this needs to be fixed.

        # Combine all features (sparse/dense, sequence-/sentence-level) into one tensor
        seq_sent_features, mask_combined_sequence_sentence = self._tf_layers[
            "feature_combining_layer"
        ]((sequence_features, sentence_features, sequence_feature_lengths))

        seq_sent_features = self._tf_layers["ffnn"](seq_sent_features, training)

        # If using masked language modeling, mask the transformer inputs and get labels
        # for the masked tokens and a boolean mask.
        if self._enables_mlm and training:
            seq_sent_features, token_ids, mlm_boolean_mask = self._create_mlm_tensors(
                sequence_features,
                seq_sent_features,
                mask_combined_sequence_sentence,
                training,
            )
        else:
            # tf.zeros((0,)) is an alternative to None
            token_ids = tf.zeros((0,))
            mlm_boolean_mask = tf.zeros((0,))

        # Apply the transformer (if present), hence reducing a sequences of features per
        # input example into a simple fixed-size embeddings.
        if "transformer" in self._tf_layers:
            mask_padding = 1 - mask_combined_sequence_sentence
            outputs, attention_weights = self._tf_layers["transformer"](
                seq_sent_features, 1 - mask_padding, training
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
