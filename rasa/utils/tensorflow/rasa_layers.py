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

# class ArtificialLayer(tf.keras.layers.Layer):
#     def __init__(self, name: Text):
#         super().__init__(name=f"{name}_artificial")

#     def call(self, arg1, arg2) -> tf.Tensor:
#         result = arg2
#         return result


class ConcatenateSparseDenseFeatures(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        attribute: Text,
        feature_type: Text,
        data_signature: List[FeatureSignature],
        dropout_rate: float,
        sparse_dropout: bool,
        dense_dropout: bool,
        dense_concat_dimension: int,
        **sparse_to_dense_kwargs: Any,
    ) -> None:
        if not data_signature:
            raise ValueError("No feature signatures found!")
        super().__init__(
            name=f"concatenate_sparse_dense_features_{attribute}_{feature_type}"
        )
        self.have_sparse_features = any(
            [signature.is_sparse for signature in data_signature]
        )
        self.have_dense_features = any(
            [not signature.is_sparse for signature in data_signature]
        )

        all_sparse_units = sum(
            [
                dense_concat_dimension
                for signature in data_signature
                if signature.is_sparse
            ]
        )
        all_dense_units = sum(
            [signature.units for signature in data_signature if not signature.is_sparse]
        )
        self.output_units = all_sparse_units + all_dense_units

        self.use_sparse_dropout = sparse_dropout
        self.use_dense_dropout = dense_dropout

        if self.have_sparse_features:
            if "name" not in sparse_to_dense_kwargs:
                sparse_to_dense_kwargs[
                    "name"
                ] = f"sparse_to_dense.{attribute}_{feature_type}"
            # print("DfS")
            self._sparse_to_dense = layers.DenseForSparse(**sparse_to_dense_kwargs)

            # print("SD")
            if self.use_sparse_dropout:
                self._sparse_dropout = layers.SparseDropout(rate=dropout_rate)

        if self.use_dense_dropout:
            # print("D")
            self._dense_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.identifier = f"{attribute}_{feature_type}"

    def call(
        self,
        inputs: Tuple[List[Union[tf.Tensor, tf.SparseTensor]]],
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> tf.Tensor:
        features = inputs[0]

        # print(f"INSIDE sparse+dense layer ({self.identifier})")
        # print(f"--> indices -> {features[0].indices}")
        # print(f"--> values -> {features[0].values}")

        dense_features = []
        for f in features:
            if isinstance(f, tf.SparseTensor):
                if self.use_sparse_dropout:
                    # print("USE DROPOUT")
                    _f = self._sparse_dropout(f, training)
                else:
                    # print("DON'T USE DROPOUT")
                    _f = f

                
                # print(f"sparse before {self.identifier} # {_f.shape} # {_f.indices} # {_f.values}")
                dense_f = self._sparse_to_dense(_f)
                # print(f"SPARSE_TO_DENSE: {self._sparse_to_dense.name} {self._sparse_to_dense.units}")
                
                # weights = self._sparse_to_dense.get_weights()
                # weights_new = [np.full(w.shape, 0.3) for w in weights]
                # self._sparse_to_dense.set_weights(weights_new)
                # dense_f = self._sparse_to_dense(_f)
                # print(f"SPARSE_TO_DENSE: {self._sparse_to_dense.name} {self._sparse_to_dense.units}")
                # for w in weights:
                #     print(f"w # {w.shape} # {w}")
                # print(f"sparse after converted into dense {self.identifier} # {dense_f.shape} # {dense_f}")


                if self.use_dense_dropout:
                    dense_f = self._dense_dropout(dense_f, training)

                # print(f"sparse after # {dense_f.shape} # {dense_f}")
                dense_features.append(dense_f)
            else:
                # print(f"dense before/after # {f.shape} # {f}")
                dense_features.append(f)

        return tf.concat(dense_features, axis=-1)


class ConcatenateSequenceSentenceFeatures(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        layer_name_suffix: Text,
        concat_dimension: int,
        sequence_signature: FeatureSignature,
        sentence_signature: FeatureSignature,
        concat_layers_kwargs: Dict[Text, Any] = {},
    ) -> None:
        super().__init__(
            name=f"concatenate_sequence_sentence_features_{layer_name_suffix}"
        )
        if sequence_signature and sentence_signature:
            self.do_concatenation = True
            if sequence_signature.units != sentence_signature.units:
                self.unify_dimensions_before_concat = True
                self.output_units = concat_dimension
                self.unify_dimensions_layers = {}
                for feature_type in [SEQUENCE, SENTENCE]:
                    if "layer_name_suffix" not in concat_layers_kwargs:
                        concat_layers_kwargs[
                            "layer_name_suffix"
                        ] = f"unify_dimensions_before_concat.{layer_name_suffix}_{feature_type}"
                    # print("FFNN")
                    self.unify_dimensions_layers[feature_type] = layers.Ffnn(
                        **concat_layers_kwargs
                    )
            else:
                self.unify_dimensions_before_concat = False
                self.output_units = sequence_signature.units
        else:
            self.do_concatenation = False
            if sequence_signature and not sentence_signature:
                self.return_just = SEQUENCE
                self.output_units = sequence_signature.units
            elif sentence_signature and not sequence_signature:
                self.return_just = SENTENCE
                self.output_units = sentence_signature.units

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        sequence = inputs[0]
        sentence = inputs[1]
        mask_text = inputs[2]

        if self.do_concatenation:
            if self.unify_dimensions_before_concat:
                sequence = self.unify_dimensions_layers[SEQUENCE](sequence)
                sentence = self.unify_dimensions_layers[SENTENCE](sentence)

            # we need to concatenate the sequence features with the sentence features
            # we cannot use tf.concat as the sequence features are padded

            # (1) get position of sentence features in mask
            last = mask_text * tf.math.cumprod(
                1 - mask_text, axis=1, exclusive=True, reverse=True
            )
            # (2) multiply by sentence features so that we get a matrix of
            #     batch-dim x seq-dim x feature-dim with zeros everywhere except for
            #     for the sentence features
            sentence = last * sentence

            # (3) add a zero to the end of sequence matrix to match the final shape
            sequence = tf.pad(sequence, [[0, 0], [0, 1], [0, 0]])

            # (4) sum up sequence features and sentence features
            return sequence + sentence
        elif self.return_just == SEQUENCE:
            return sequence
        elif self.return_just == SENTENCE:
            return sentence


# does:
# 1. sparse+dense
# 2. seq+sent
class RasaInputLayer(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        name: Text,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        if not data_signature or not (
            len(data_signature.get(SENTENCE, [])) > 0
            or len(data_signature.get(SEQUENCE, [])) > 0
        ):
            raise ValueError("The data signature must contain some features.")

        super().__init__(name=f"rasa_input_layer_{name}")
        # SPARSE + DENSE
        self.concat_sparse_dense = {}
        for feature_type in [SENTENCE, SEQUENCE]:
            if feature_type in data_signature and data_signature[feature_type]:
                sparse_to_dense_layer_options = {
                    "units": config[DENSE_DIMENSION][name],
                    "reg_lambda": config[REGULARIZATION_CONSTANT],
                    "name": f"sparse_to_dense.{name}_{feature_type}",
                }
                self.concat_sparse_dense[feature_type] = ConcatenateSparseDenseFeatures(
                    attribute=name,
                    feature_type=feature_type,
                    data_signature=data_signature.get(feature_type, []),
                    dropout_rate=config[DROP_RATE],
                    sparse_dropout=config[SPARSE_INPUT_DROPOUT],
                    dense_dropout=config[DENSE_INPUT_DROPOUT],
                    dense_concat_dimension=config[DENSE_DIMENSION][name],
                    **sparse_to_dense_layer_options,
                )
            else:
                self.concat_sparse_dense[feature_type] = None

        # SEQUENCE + SENTENCE
        self.do_seq_sent_concat = all(
            [
                len(data_signature.get(feature_type, [])) > 0
                for feature_type in [SEQUENCE, SENTENCE]
            ]
        )
        if self.do_seq_sent_concat:
            seq_sent_data_signatures = {}
            for feature_type in [SEQUENCE, SENTENCE]:
                signature_existing = data_signature[feature_type][0]
                signature_new = FeatureSignature(
                    is_sparse=False,
                    units=self.concat_sparse_dense[feature_type].output_units,
                    number_of_dimensions=signature_existing.number_of_dimensions,
                )
                seq_sent_data_signatures[feature_type] = signature_new

            concat_layers_kwargs = {
                "layer_sizes": [config[CONCAT_DIMENSION][name]],
                "dropout_rate": config[DROP_RATE],
                "reg_lambda": config[REGULARIZATION_CONSTANT],
                "sparsity": config[WEIGHT_SPARSITY],
            }

            self.concat_seq_sent = ConcatenateSequenceSentenceFeatures(
                sequence_signature=seq_sent_data_signatures[SEQUENCE],
                sentence_signature=seq_sent_data_signatures[SENTENCE],
                concat_dimension=config[CONCAT_DIMENSION].get(name, None),
                concat_layers_kwargs=concat_layers_kwargs,
                layer_name_suffix=name,
            )

        if self.do_seq_sent_concat:
            self.output_units = self.concat_seq_sent.output_units
        elif self.concat_sparse_dense[SEQUENCE]:
            self.output_units = self.concat_sparse_dense[SEQUENCE].output_units
        else:
            self.output_units = self.concat_sparse_dense[SENTENCE].output_units

        self.identifier = name

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
        sequence_features = inputs[0]
        sentence_features = inputs[1]
        mask_sequence = inputs[2]
        mask_text = inputs[3]

        # print(f"INSIDE sent+seq layer {self.identifier}")
        # for (f, t) in [(sequence_features, "sequence"), (sentence_features, "sentence")]:
        #     print(f"--> {t}.indices -> {f[0].indices}")
        #     print(f"--> {t}.values -> {f[0].values}")
        # print(len(sequence_features), len(sentence_features))


        if self.do_seq_sent_concat:
            _inputs1 = (sequence_features,)
            sequence = self.concat_sparse_dense[SEQUENCE](_inputs1, training=training)
            # print(f"sequence before mask # {sequence.shape} # {sequence}")

            if sequence is not None and mask_sequence is not None:
                sequence = sequence * mask_sequence

            _inputs2 = (sentence_features,)
            sentence = self.concat_sparse_dense[SENTENCE](_inputs2, training=training)
            # print(f"sequence AFTER SPARSE+DENSE # {sequence.shape} # {sequence}")
            # print(f"sentence AFTER SPARSE+DENSE # {sentence.shape} # {sentence}")

            # exit(0)
            _inputs3 = (sequence, sentence, mask_text)
            sequence_sentence = self.concat_seq_sent(_inputs3)

            return sequence_sentence

        elif self.concat_sparse_dense[SEQUENCE]:
            _inputs = (sequence_features,)
            sequence = self.concat_sparse_dense[SEQUENCE](_inputs, training=training)

            return sequence
        else:
            _inputs = (sentence_features,)
            sentence = self.concat_sparse_dense[SENTENCE](_inputs, training=training)

            return sentence


# does:
# 1. input_layer
# 2. ffnn
# [3. MLM: masking & creating dense labels to sample from]
# 4. transformer
class RasaSequenceLayer(tf.keras.layers.Layer):
    # TODO add docstring
    def __init__(
        self,
        name: Text,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
    ) -> None:
        super().__init__(name=f"rasa_input_layer_{name}")
        self.config = config

        # RASA INPUT LAYER
        self.input_layer = RasaInputLayer(name, data_signature, config)

        # FFNN
        # print("FFNN(seqlayer)")
        self.ffnn = layers.Ffnn(
            config[HIDDEN_LAYERS_SIZES][name],
            config[DROP_RATE],
            config[REGULARIZATION_CONSTANT],
            config[WEIGHT_SPARSITY],
            layer_name_suffix=name,
        )

        # MLM
        # for sequential text features prepare the logic for producing dense token embeddings
        # to be used as labels in MLM. these will be sampled from for negative sampling.
        if name == TEXT and SEQUENCE in data_signature:
            # print("InpMask")
            self.input_mask_layer = layers.InputMask()

            self.produce_dense_token_ids = True
            has_sparse = any(
                [signature.is_sparse for signature in data_signature[SEQUENCE]]
            )
            has_dense = any(
                [not signature.is_sparse for signature in data_signature[SEQUENCE]]
            )
            # if dense features are present, we use those as unique token-level embeddings,
            # otherwise we create these from the sparse features by using a simple layer.
            if has_sparse and not has_dense:
                # print("DFS-ids")
                self.sparse_to_dense_token_ids = layers.DenseForSparse(
                    units=2,
                    use_bias=False,
                    trainable=False,
                    name=f"sparse_to_dense_token_ids.{name}",
                )
        else:
            self.produce_dense_token_ids = False

        # TRANSFORMER
        num_layers = config[NUM_TRANSFORMER_LAYERS]
        if isinstance(num_layers, dict):
            num_layers = num_layers[name]
        size = config[TRANSFORMER_SIZE]
        if isinstance(size, dict):
            size = size[name]
        self.num_transformer_layers = num_layers
        self.transformer_size = size

        if self.num_transformer_layers > 0:
            self.transformer = TransformerEncoder(
                num_layers=self.num_transformer_layers,
                units=self.transformer_size,
                num_heads=config[NUM_HEADS],
                filter_units=size * 4,
                reg_lambda=config[REGULARIZATION_CONSTANT],
                dropout_rate=config[DROP_RATE],
                attention_dropout_rate=config[DROP_RATE_ATTENTION],
                sparsity=config[WEIGHT_SPARSITY],
                unidirectional=config[UNIDIRECTIONAL_ENCODER],
                use_key_relative_position=config[KEY_RELATIVE_ATTENTION],
                use_value_relative_position=config[VALUE_RELATIVE_ATTENTION],
                max_relative_position=config[MAX_RELATIVE_POSITION],
                name=f"{name}_encoder",
            )

        # TODO: should this simply use NUM_TRANSFORMER_LAYERS?
        # if config[f"{DIALOGUE}_{NUM_TRANSFORMER_LAYERS}"] > 0:
        if self.num_transformer_layers > 0:
            self.output_units = self.transformer_size
        elif config[HIDDEN_LAYERS_SIZES][name]:
            self.output_units = config[HIDDEN_LAYERS_SIZES][name][-1]
        else:
            self.output_units = self.input_layer.output_units

    def _features_as_seq_ids(
        self, features: List[Union[tf.Tensor, tf.SparseTensor]]
    ) -> Optional[tf.Tensor]:
        """Creates dense labels for negative sampling."""

        # if there are dense features - we can use them
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                seq_ids = tf.stop_gradient(f)
                # add a zero to the seq dimension for the sentence features
                seq_ids = tf.pad(seq_ids, [[0, 0], [0, 1], [0, 0]])
                return seq_ids

        # use additional sparse to dense layer
        for f in features:
            if isinstance(f, tf.SparseTensor):
                seq_ids = tf.stop_gradient(self.sparse_to_dense_token_ids(f))
                # add a zero to the seq dimension for the sentence features
                seq_ids = tf.pad(seq_ids, [[0, 0], [0, 1], [0, 0]])
                return seq_ids

        return None

    def call(
        self,
        inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
            tf.Tensor,
            # bool,
        ],
        masked_lm_loss: bool = False,
        training: Optional[Union[tf.Tensor, bool]] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        sequence_features = inputs[0]
        sentence_features = inputs[1]
        mask_sequence = inputs[2]
        mask_text = inputs[3]

        # print("INSIDE sequence_layer")
        # for (f, t) in [(sequence_features, "sequence"), (sentence_features, "sentence")]:
        #     print(f"--> {t}.indices -> {f[0].indices}")
        #     print(f"--> {t}.values -> {f[0].values}")


        _inputs = (sequence_features, sentence_features, mask_sequence, mask_text)
        x = self.input_layer(_inputs)
        # print(f"x after input layer # {x.shape} # {x}")

        x = self.ffnn(x, training)
        # print(f"x after ffnn # {x.shape} # {x}")

        if self.produce_dense_token_ids:
            seq_ids = self._features_as_seq_ids(sequence_features)
        else:
            seq_ids = None

        # TODO unify this with self.produce_dense_token_ids?
        if masked_lm_loss:
            transformer_inputs, lm_mask_bool = self.input_mask_layer(
                x, mask_text, training
            )
        else:
            transformer_inputs = x
            lm_mask_bool = None

        if self.num_transformer_layers > 0:
            outputs = self.transformer(transformer_inputs, 1 - mask_text, training)
            outputs = tfa.activations.gelu(outputs)
        else:
            outputs = transformer_inputs

        return outputs, x, seq_ids, lm_mask_bool
