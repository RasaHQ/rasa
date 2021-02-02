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
        data_signature: A list of `FeatureSignature`s for the given attribute and
            feature type.
        dropout_rate: Float between 0 and 1; fraction of the sparse tensors' units to
            be dropped if `sparse_dropout` or `dense_dropout` is enabled.
        sparse_dropout: Boolean; whether to apply dropout to sparse tensors before
            converting them into dense ones.
        dense_dropout: Boolean; whether to apply dropout to sparse tensors after
            converting them into dense ones.
        sparse_to_dense_units: Positive integer; sparse tensors will be converted into
            dense tensors with last dimension of this size.
        **sparse_to_dense_kwargs: Additional arguments to pass to the constructor of the
            DenseForSparse layer (used to convert sparse tensors to dense).

    Input shape:
        List of N-D tensors, each with shape: `(batch_size, ..., input_dim)`.
        All tensors must have the same shape, except the last dimension.
        All sparse tensors must have the same shape including the last dimension.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., units)` where `units` is the sum of
        the last dimension sizes across all input sensors, with sparse tensors instead
        contributing `sparse_to_dense_units` units each.

    Raises:
        A ValueError if no feature signatures are provided.
    """

    def __init__(
        self,
        attribute: Text,
        feature_type: Text,
        data_signature: List[FeatureSignature],
        dropout_rate: float,
        sparse_dropout: bool,
        dense_dropout: bool,
        sparse_to_dense_units: int,
        **sparse_to_dense_kwargs: Dict[Text, Any],
    ) -> None:
        super().__init__(
            name=f"concatenate_sparse_dense_features_{attribute}_{feature_type}"
        )

        if not data_signature:
            raise ValueError("No feature signatures found!")

        # compute the output units from the provided data signature
        output_units_from_dense = sum(
            [signature.units for signature in data_signature if not signature.is_sparse]
        )
        output_units_from_sparse = sum(
            [
                sparse_to_dense_units
                for signature in data_signature
                if signature.is_sparse
            ]
        )
        self.output_units = output_units_from_dense + output_units_from_sparse

        # prepare dropout and sparse-to-dense layers if any sparse tensors are expected
        if any([signature.is_sparse for signature in data_signature]):
            sparse_to_dense_kwargs[
                "name"
            ] = f"sparse_to_dense.{attribute}_{feature_type}"
            sparse_to_dense_kwargs["units"] = sparse_to_dense_units
            self._sparse_to_dense = layers.DenseForSparse(**sparse_to_dense_kwargs)

            if sparse_dropout:
                self._sparse_dropout = layers.SparseDropout(rate=dropout_rate)

            if dense_dropout:
                # TODO: check that this really needs to happen only for sparse features
                # and not for all!
                self._dense_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(
        self,
        inputs: Tuple[List[Union[tf.Tensor, tf.SparseTensor]]],
        training: Optional[Union[tf.Tensor, bool]] = None,
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
                if hasattr(self, "_sparse_dropout"):
                    f = self._sparse_dropout(f, training)

                f = self._sparse_to_dense(f)

                if hasattr(self, "_dense_dropout"):
                    f = self._dense_dropout(f, training)

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
        concat_dimension: A positive integer -- number of output units of the dense
            layers used to unify the last dimension of each feature type.
        data_signature_sequence: Feature signature describing the sequence-level features.
        data_signature_sentence: Feature signature describing the sentence-level features.
        **unify_concat_dimension_kwargs: Additional arguments to pass to the constructor
            of the dense Ffnn layers for unifying the features' last dimension size.

    Input shape:
        Tuple of three 3-D dense tensors, with shapes:
            sequence: `(batch_size, max_seq_length, input_dim_seq)`
            sentence: `(batch_size, 1, input_dim_sent)`
            mask: `(batch_size, max_seq_length+1, 1)`

    Output shape:
        3-D tensor with shape: `(batch_size, sequence_length, output_dim)` where:
        - `output_dim` matches `input_dim_seq` if that is the same as `input_dim_sent`,
            otherwise `output_dim` = `concat_dimension`.
        - `sequence_length` is the sum of 2nd dimension sizes of arguments `sequence`
            and`sentence`.
    """

    def __init__(
        self,
        attribute: Text,
        concat_dimension: int,
        data_signature_sequence: FeatureSignature,
        data_signature_sentence: FeatureSignature,
        **unify_concat_dimension_kwargs: Dict[Text, Any],
    ) -> None:
        super().__init__(name=f"concatenate_sequence_sentence_features_{attribute}")

        # main use case: both sequence and sentence features are expected
        if data_signature_sequence and data_signature_sentence:
            self.do_concatenation = True

            # prepare dimension unifying layers if needed
            if data_signature_sequence.units != data_signature_sentence.units:
                self.unify_dimensions_before_concat = True
                self.output_units = concat_dimension
                self.unify_dimensions_layers = {}

                for feature_type in [SEQUENCE, SENTENCE]:
                    if "layer_name_suffix" not in unify_concat_dimension_kwargs:
                        unify_concat_dimension_kwargs[
                            "layer_name_suffix"
                        ] = f"unify_dim_before_seq_sent_concat.{attribute}_{feature_type}"
                    self.unify_dimensions_layers[feature_type] = layers.Ffnn(
                        **unify_concat_dimension_kwargs
                    )
            else:
                self.unify_dimensions_before_concat = False
                self.output_units = data_signature_sequence.units

        # edge cases where only sequence or only sentence features are expected
        else:
            self.do_concatenation = False
            if data_signature_sequence and not data_signature_sentence:
                self.only_return = SEQUENCE
                self.output_units = data_signature_sequence.units
            elif data_signature_sentence and not data_signature_sequence:
                self.only_return = SENTENCE
                self.output_units = data_signature_sentence.units

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

        if self.do_concatenation:
            if self.unify_dimensions_before_concat:
                sequence = self.unify_dimensions_layers[SEQUENCE](sequence)
                sentence = self.unify_dimensions_layers[SENTENCE](sentence)

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

        elif self.only_return == SEQUENCE:
            return sequence

        else:
            return sentence


class RasaInputLayer(tf.keras.layers.Layer):
    """Combines multiple dense or sparse feature tensors into one.

    This layer combines features by following these steps:
    1. Apply a `ConcatenateSparseDenseFeatures` layer separately to sequence- and
        sentence-level features, yielding two tensors (one for each feature type).
    2. Apply a `ConcatenateSequenceSentenceFeatures` layer to the two tensors to combine
        sequence- and sentence-level features into one tensor.

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        data_signature: A dictionary containing two lists of `FeatureSignature`s, one
            for each feature type of the given attribute.
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
            mask_sequence_sentence: dense 3-D tensor with the shape `(batch_size, max_
                sequence_length+1, 1)`, i.e. with the 2nd dimension combining the lengths
                of sequence- and sentence-level features.

    Output shape:
        3-D tensor with shape: `(batch_size, sequence_length, output_dim)` where
        `output_dim` is completely  determined by the internally applied
        `ConcatenateSparseDenseFeatures` layer and `sequence_length` is completely
        determined by the internally applied `ConcatenateSequenceSentenceFeatures` layer.

    Raises:
        A ValueError if no feature signatures are provided.
    """

    def __init__(
        self,
        attribute: Text,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        if not data_signature or not (
            len(data_signature.get(SENTENCE, [])) > 0
            or len(data_signature.get(SEQUENCE, [])) > 0
        ):
            raise ValueError("The data signature must contain some features.")

        super().__init__(name=f"rasa_input_layer_{attribute}")

        # 1. prepare layers for combining sparse and dense features
        self.layers_sparse_dense = {}
        for feature_type in [SENTENCE, SEQUENCE]:
            # prepare the layer only if some feature signature are present for this
            # feature type
            if feature_type in data_signature and data_signature[feature_type]:
                sparse_to_dense_layer_options = {
                    "reg_lambda": config[REGULARIZATION_CONSTANT],
                }
                self.layers_sparse_dense[feature_type] = ConcatenateSparseDenseFeatures(
                    attribute=attribute,
                    feature_type=feature_type,
                    data_signature=data_signature.get(feature_type, []),
                    dropout_rate=config[DROP_RATE],
                    sparse_dropout=config[SPARSE_INPUT_DROPOUT],
                    dense_dropout=config[DENSE_INPUT_DROPOUT],
                    sparse_to_dense_units=config[DENSE_DIMENSION][attribute],
                    **sparse_to_dense_layer_options,
                )
            else:
                self.layers_sparse_dense[feature_type] = None

        # 2. prepare layer for combining sequence- and sentence-level features
        self.have_all_feature_types = all(
            [
                len(data_signature.get(feature_type, [])) > 0
                for feature_type in [SEQUENCE, SENTENCE]
            ]
        )
        # prepare the combining layer only if all two feature types are expected, otherwise
        # there's nothing to combine
        if self.have_all_feature_types:

            # Construct new feature signatures to be consumed by the combining layer.
            # These differ from the originally received signatures: the new ones reflect
            # that the sparse+dense combining layer will have been applied to features.
            new_signatures = {}
            for feature_type in [SEQUENCE, SENTENCE]:
                # it's enough to take any signature for the given feature type, we take
                # the first one
                original_signature = data_signature[feature_type][0]

                # update attributes to take into account the effect of the sparse+dense
                # combining layer
                new_signature = FeatureSignature(
                    is_sparse=False,
                    units=self.layers_sparse_dense[feature_type].output_units,
                    number_of_dimensions=original_signature.number_of_dimensions,
                )
                new_signatures[feature_type] = new_signature

            # prepare the combining layer itself
            unify_concat_dimension_kwargs = {
                "layer_sizes": [config[CONCAT_DIMENSION][attribute]],
                "dropout_rate": config[DROP_RATE],
                "reg_lambda": config[REGULARIZATION_CONSTANT],
                "sparsity": config[WEIGHT_SPARSITY],
            }
            self.concat_seq_sent = ConcatenateSequenceSentenceFeatures(
                attribute=attribute,
                data_signature_sequence=new_signatures[SEQUENCE],
                data_signature_sentence=new_signatures[SENTENCE],
                concat_dimension=config[CONCAT_DIMENSION].get(attribute, None),
                **unify_concat_dimension_kwargs,
            )

        # if only one feature type is present, prepare for only combining sparse & dense
        # features
        if self.have_all_feature_types:
            self.output_units = self.concat_seq_sent.output_units
        elif self.layers_sparse_dense[SEQUENCE]:
            self.output_units = self.layers_sparse_dense[SEQUENCE].output_units
        else:
            self.output_units = self.layers_sparse_dense[SENTENCE].output_units

    def call(
        self,
        inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
            tf.Tensor,
        ],
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> tf.Tensor:
        """Combine multiple 3-D dense/sparse feature tensors into one.

        Arguments:
            inputs: Tuple containing:
                sequence_features: List of 3-D dense or sparse tensors with token-level
                    features.
                sentence_features: List of 3-D dense or sparse tensors with sentence-level
                    features.
                mask_sequence: a 3-D tensor mask that has 1s at real and 0s at padded
                    positions corresponding to tokens in `sequence_features`.
                mask_sequence_sentence: a 3-D tensor mask similar to `mask_sequence` but
                    having each sequence of 1s longer by 1 to account for sequence 
                    lengths of sequence- and sentence-level features being combined.
            training: Python boolean indicating whether the layer should behave in
                training mode (applying dropout to sparse tensors if applicable) or in
                inference mode (not applying dropout).

        Returns:
            Single dense 3-D tensor containing the combined sequence- and sentence-
            level, sparse & dense features.
        """
        sequence_features = inputs[0]
        sentence_features = inputs[1]
        mask_sequence = inputs[2]
        mask_sequence_sentence = inputs[3]

        # different feature types are present, make them dense & combine them
        if self.have_all_feature_types:
            _inputs = (sequence_features,)
            sequence = self.layers_sparse_dense[SEQUENCE](_inputs, training=training)

            # apply mask which has 1s at positions of real tokens and 0s at all padded
            # token positions. This is needed because the sparse+dense combining layer
            # might've turned some fake (padded) features (i.e. 0s) into non-zero numbers
            # and we want those to become zeros again.
            # This step isn't needed for sentence-level features because those are never
            # padded -- the effective sequence length in their case is always 1.
            if mask_sequence is not None:
                sequence = sequence * mask_sequence

            _inputs = (sentence_features,)
            sentence = self.layers_sparse_dense[SENTENCE](_inputs, training=training)

            _inputs = (sequence, sentence, mask_sequence_sentence)
            sequence_sentence = self.concat_seq_sent(_inputs)

            return sequence_sentence

        # only one feature type is present - make it dense but skip combining
        elif self.layers_sparse_dense[SEQUENCE]:
            _inputs = (sequence_features,)
            sequence = self.layers_sparse_dense[SEQUENCE](_inputs, training=training)

            return sequence
        else:
            _inputs = (sentence_features,)
            sentence = self.layers_sparse_dense[SENTENCE](_inputs, training=training)

            return sentence


class RasaSequenceLayer(tf.keras.layers.Layer):
    """Creates an embedding from all features for a sequence attribute; facilitates MLM.

    This layer combines all features for an attribute and embeds them using a 
    transformer. The layer is meant only for attributes with sequence-level features,
    such as `text` and `action_text`.

    Internally, this layer extends RasaInputLayer and goes through the following steps:
    1. Combine features using RasaInputLayer.
    2. Apply a dense layer(s) to the combined features.
    3. Optionally (during training for the `text` attribute), apply masking to the
        features and create further helper variables for masked language modeling.
    4. Embed the features using a transformer, effectively going from variable-length
        sequences of features to fixed-size input example embeddings.

    Arguments:
        attribute: Name of attribute (e.g. `text` or `label`) whose features will be
            processed.
        data_signature: A dictionary containing two lists of `FeatureSignature`s, one
            for each feature type of the given attribute.
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
            mask_sequence_sentence: dense 3-D tensor with the shape `(batch_size, max_
                sequence_length+1, 1)`, i.e. with the 2nd dimension combining the lengths
                of sequence- and sentence-level features.

    Output shape:
        outputs: `(batch_size, max_seq_length+1, output_size)` where `output_size` matches
            the underlying transformer's output size if the transformer has some layers,
            otherwise `output_size` matches that of the Ffnn block applied to the
            combined features, or it's the output size of the underlying `RasaInputLayer`
            the Ffnn block has 0 layers. `max_seq_length` is the length of the longest
            sequence of tokens in the given batch.
        seq_sent_features: `(batch_size, max_seq_length+1, hidden_dim)`, where 
            `hidden_dim` is the output size of the underlying Ffnn block, or the output
            size of the underlying `RasaInputLayer` if the Ffnn block has 0 layers.
        token_ids: `(batch_size, max_seq_length+1, id_dim)` where id_dim is unimportant
            and it's the last-dimension size of the first sequence-level dense feature
            if any is present, and 2 otherwise. `None` if not doing MLM.
        mlm_mask_bool: `(batch_size, max_seq_length+1, 1)`, `None` if not doing MLM.
        attention_weights: `(num_transformer_layers, batch_size, num_transformer_heads, 
            max_seq_length+1, max_seq_length+1)`, `None` if the transformer has 0 layers.

    Raises:
        A ValueError if no feature signatures for sequence-level features are provided.
    """

    def __init__(
        self,
        attribute: Text,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        if not data_signature or len(data_signature.get(SEQUENCE, [])) == 0:
            raise ValueError(
                "The data signature must contain some sequence-level features but none\
                were found."
            )

        super().__init__(name=f"rasa_input_layer_{attribute}")

        self.input_layer = RasaInputLayer(attribute, data_signature, config)

        self.ffnn = layers.Ffnn(
            config[HIDDEN_LAYERS_SIZES][attribute],
            config[DROP_RATE],
            config[REGULARIZATION_CONSTANT],
            config[WEIGHT_SPARSITY],
            layer_name_suffix=attribute,
        )

        # Prepare masking and computing helper variables for masked language modeling.
        # This is only done for the text attribute and only if sequence-level (token-
        # level) features are present (MLM requires token-level information).
        if attribute == TEXT and SEQUENCE in data_signature:
            self.enables_mlm = True
            self.mlm_input_mask_layer = layers.InputMask()

            # Unique IDs of different token types are needed to construct the possible
            # label space for MLM. If dense features are present, they're used as such
            # IDs, othwerise sparse features are embedded by a non-trainable
            # DenseForSparse layer to create small embeddings that serve as IDs.
            expect_dense_seq_features = any(
                [not signature.is_sparse for signature in data_signature[SEQUENCE]]
            )
            if not expect_dense_seq_features:
                self.sparse_to_dense_token_ids = layers.DenseForSparse(
                    units=2,
                    use_bias=False,
                    trainable=False,
                    name=f"sparse_to_dense_token_ids.{attribute}",
                )
        else:
            self.enables_mlm = False

        # Prepare the transformer
        self.num_transformer_layers = config[NUM_TRANSFORMER_LAYERS]
        if isinstance(self.num_transformer_layers, dict):
            self.num_transformer_layers = self.num_transformer_layers[attribute]
        self.transformer_size = config[TRANSFORMER_SIZE]
        if isinstance(self.transformer_size, dict):
            self.transformer_size = self.transformer_size[attribute]
        if self.num_transformer_layers > 0:
            self.transformer = TransformerEncoder(
                num_layers=self.num_transformer_layers,
                units=self.transformer_size,
                num_heads=config[NUM_HEADS],
                filter_units=self.transformer_size * 4,
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

        # The output size of this layer depends on which component is the last one in
        # the pipeline because not all components (e.g. the transformer) are necessarily
        # created.
        if self.num_transformer_layers > 0:
            self.output_units = self.transformer_size
        elif len(config[HIDDEN_LAYERS_SIZES][attribute]) > 0:
            # this is the output size of (the last layer of) the Ffnn block
            self.output_units = config[HIDDEN_LAYERS_SIZES][attribute][-1]
        else:
            self.output_units = self.input_layer.output_units

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
                return tf.stop_gradient(self.sparse_to_dense_token_ids(f))

    def call(
        self,
        inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
            tf.Tensor,
        ],
        masked_lm_loss: bool = False,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[
        tf.Tensor,
        tf.Tensor,
        Optional[tf.Tensor],
        Optional[tf.Tensor],
        Optional[tf.Tensor],
    ]:
        """Combine all features for an attribute into one and embed using a transformer.

        Arguments:
            inputs: Tuple containing:
                sequence_features: List of 3-D dense or sparse tensors with token-level
                    features.
                sentence_features: List of 3-D dense or sparse tensors with sentence-level
                    features.
                mask_sequence: a 3-D tensor mask that has 1s at real and 0s at padded
                    positions corresponding to tokens in `sequence_features`.
                mask_sequence_sentence: a 3-D tensor mask similar to `mask_sequence` but
                    having each sequence of 1s longer by 1 to account for sequence 
                    lengths of sequence- and sentence-level features being combined.
            masked_lm_loss: Boolean, using True means doing masked language modeling.
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
                `None` if not doing MLM.
            mlm_mask_bool: 3-D tensor mask that has 1s where real tokens in `outputs` 
                were masked and 0s elsewhere. `None` if not doing MLM.
            attention_weights: 5-D tensor containing self-attention weights received 
                from the underlying transformer. `None` if the transformer has 0 layers.

        Raises:
            ValueError if the layer wasn't prepared for MLM but masked_lm_loss is True.
        """
        if masked_lm_loss and not self.enables_mlm:
            raise ValueError(
                f"This layer wasn't prepared to do masked language modeling, hence you\
                cannot call it with masked_lm_loss=True. MLM is currently supported only\
                with sequence-level features for the '{TEXT}' attribute."
            )

        sequence_features = inputs[0]
        sentence_features = inputs[1]
        mask_sequence = inputs[2]
        mask_sequence_sentence = inputs[3]

        # Combine all features (sparse/dense, sequence-/sentence-level) into one tensor
        _inputs = (
            sequence_features,
            sentence_features,
            mask_sequence,
            mask_sequence_sentence,
        )
        seq_sent_features = self.input_layer(_inputs)

        seq_sent_features = self.ffnn(seq_sent_features, training)

        # Produce helper variables for masked language modelling if needed
        if self.enables_mlm and masked_lm_loss:
            # token_ids are embeddings with the same shape as the first dense sequence
            # feature or alternatively the last dimension will have size 2 if there are
            # only sparse sequence features present. In any case, these embeddings can
            # be viewed as labels/unique IDs of all input tokens (to be used later)
            # because these embeddings aren't affected by dropout or masking and are
            # effectively always unique for different input tokens (and same for the
            # same tokens).
            token_ids = self._features_as_token_ids(sequence_features)
            # Pad in the sequence dimension to match the shape of combined sequence- and
            # sentence-level features (sentence-level features effectively have sequence
            # length of 1).
            token_ids = tf.pad(token_ids, [[0, 0], [0, 1], [0, 0]])

            # mlm_mask_bool has the same shape as mask_sequence_sentence (i.e. as the
            # tensor with all combined features), with True meaning tokens that are
            # masked and False meaning tokens that aren't masked or that are fake
            # (padded) tokens.
            transformer_inputs, mlm_mask_bool = self.mlm_input_mask_layer(
                seq_sent_features, mask_sequence_sentence, training
            )
        else:
            token_ids = None
            transformer_inputs = seq_sent_features
            mlm_mask_bool = None

        # Apply the transformer (if present), hence reducing a sequences of features per
        # input example into a simple fixed-size embeddings.
        if self.num_transformer_layers > 0:
            mask_padding = 1 - mask_sequence_sentence
            outputs, attention_weights = self.transformer(
                transformer_inputs, 1 - mask_padding, training
            )
            outputs = tfa.activations.gelu(outputs)
        else:
            outputs, attention_weights = transformer_inputs, None

        return outputs, seq_sent_features, token_ids, mlm_mask_bool, attention_weights
