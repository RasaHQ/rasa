---
sidebar_label: rasa.utils.tensorflow.rasa_layers
title: rasa.utils.tensorflow.rasa_layers
---
## RasaCustomLayer Objects

```python
class RasaCustomLayer(tf.keras.layers.Layer)
```

Parent class for all classes in `rasa_layers.py`.

Allows a shared implementation for adjusting `DenseForSparse`
layers during incremental training.

During fine-tuning, sparse feature sizes might change due to addition of new data.
If this happens, we need to adjust our `DenseForSparse` layers to a new size.
`ConcatenateSparseDenseFeatures`, `RasaSequenceLayer` and
`RasaFeatureCombiningLayer` all inherit from `RasaCustomLayer` and thus can
change their own `DenseForSparse` layers if it&#x27;s needed.

#### adjust\_sparse\_layers\_for\_incremental\_training

```python
def adjust_sparse_layers_for_incremental_training(new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]], old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]], reg_lambda: float) -> None
```

Finds and adjusts `DenseForSparse` layers during incremental training.

Recursively looks through the layers until it finds all the `DenseForSparse`
ones and adjusts those which have their sparse feature sizes increased.

This function heavily relies on the name of `DenseForSparse` layer being
in the following format - f&quot;sparse_to_dense.{attribute}_{feature_type}&quot; -
in order to correctly extract the attribute and feature type.

New and old sparse feature sizes could look like this:
{TEXT: {FEATURE_TYPE_SEQUENCE: [4, 24, 128], FEATURE_TYPE_SENTENCE: [4, 128]}}

**Arguments**:

- `new_sparse_feature_sizes` - sizes of current sparse features.
- `old_sparse_feature_sizes` - sizes of sparse features the model was
  previously trained on.
- `reg_lambda` - regularization constant.

## ConcatenateSparseDenseFeatures Objects

```python
class ConcatenateSparseDenseFeatures(RasaCustomLayer)
```

Combines multiple sparse and dense feature tensors into one dense tensor.

This layer combines features from various featurisers into a single feature array
per input example. All features must be of the same feature type, i.e. sentence-
level or sequence-level (token-level).

The layer combines a given list of tensors (whether sparse or dense) by:
1. converting sparse tensors into dense ones
2. optionally, applying dropout to sparse tensors before and/or after the conversion
3. concatenating all tensors along the last dimension

**Arguments**:

- `attribute` - Name of attribute (e.g. `text` or `label`) whose features will be
  processed.
- `feature_type` - Feature type to be processed -- `sequence` or `sentence`.
- `feature_type_signature` - A list of signatures for the given attribute and feature
  type.
- `config` - A model config for correctly parametrising the layer.
  
  Input shape:
  Tuple containing one list of N-D tensors, each with shape: `(batch_size, ...,
  input_dim)`.
  All dense tensors must have the same shape, except possibly the last dimension.
  All sparse tensors must have the same shape, including the last dimension.
  
  Output shape:
  N-D tensor with shape: `(batch_size, ..., units)` where `units` is the sum of
  the last dimension sizes across all input tensors, with sparse tensors instead
  contributing `config[DENSE_DIMENSION][attribute]` units each.
  

**Raises**:

  A `TFLayerConfigException` if no feature signatures are provided.
  

**Attributes**:

- `output_units` - The last dimension size of the layer&#x27;s output.

#### \_\_init\_\_

```python
def __init__(attribute: Text, feature_type: Text, feature_type_signature: List[FeatureSignature], config: Dict[Text, Any]) -> None
```

Creates a new `ConcatenateSparseDenseFeatures` object.

#### call

```python
def call(inputs: Tuple[List[Union[tf.Tensor, tf.SparseTensor]]], training: bool = False) -> tf.Tensor
```

Combines sparse and dense feature tensors into one tensor.

**Arguments**:

- `inputs` - Contains the input tensors, all of the same rank.
- `training` - A flag indicating whether the layer should behave in training mode
  (applying dropout to sparse tensors if applicable) or in inference mode
  (not applying dropout).
  

**Returns**:

  Single tensor with all input tensors combined along the last dimension.

## RasaFeatureCombiningLayer Objects

```python
class RasaFeatureCombiningLayer(RasaCustomLayer)
```

Combines multiple dense or sparse feature tensors into one.

This layer combines features by following these steps:
1. Apply a `ConcatenateSparseDenseFeatures` layer separately to sequence- and
sentence-level features, yielding two tensors (one for each feature type).
2. Concatenate the sequence- and sentence-level tensors along the sequence dimension
by appending sentence-level features at the first available token position after
the sequence-level (token-level) features.

**Arguments**:

- `attribute` - Name of attribute (e.g. `text` or `label`) whose features will be
  processed.
- `attribute_signature` - A dictionary containing two lists of feature signatures,
  one for each feature type (`sequence` or `sentence`) of the given attribute.
- `config` - A model config used for correctly parameterising the layer and the
  `ConcatenateSparseDenseFeatures` layer it uses internally.
  
  Input shape:
  Tuple of three input tensors:
- `sequence_features` - List of 3-D dense or sparse tensors, each with shape
  `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
  different for sparse vs dense tensors. See the input shape of
  `ConcatenateSparseDenseFeatures` for more information.
- `sentence_features` - List of 3-D dense or sparse tensors, each with shape
  `(batch_size, 1, input_dim)` where `input_dim` can be different for
  sparse vs dense tensors, and can differ from that in
  `sequence_features`. See the input shape of
  `ConcatenateSparseDenseFeatures` for more information.
- `sequence_feature_lengths` - Dense tensor of shape `(batch_size, )`.
  
  Output shape:
- `combined_features` - A 3-D tensor with shape `(batch_size, sequence_length,
  units)` where `units` is  completely  determined by the internally applied
  `ConcatenateSparseDenseFeatures` layer and `sequence_length` is the combined
  length of sequence- and sentence-level features: `max_seq_length + 1` if
  both feature types are present, `max_seq_length` if only sequence-level
  features are present, and 1 if only sentence-level features are present).
- `mask_combined_sequence_sentence` - A 3-D tensor with shape
  `(batch_size, sequence_length, 1)`.
  

**Raises**:

  A `TFLayerConfigException` if no feature signatures are provided.
  

**Attributes**:

- `output_units` - The last dimension size of the layer&#x27;s `combined_features` output.

#### \_\_init\_\_

```python
def __init__(attribute: Text, attribute_signature: Dict[Text, List[FeatureSignature]], config: Dict[Text, Any]) -> None
```

Creates a new `RasaFeatureCombiningLayer` object.

#### call

```python
def call(inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
        ], training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]
```

Combines multiple 3-D dense/sparse feature tensors into one.

**Arguments**:

- `inputs` - Tuple containing:
- `sequence_features` - Dense or sparse tensors representing different
  token-level features.
- `sentence_features` - Dense or sparse tensors representing sentence-level
  features.
- `sequence_feature_lengths` - A tensor containing the real sequence length
  (the number of real -- not padding -- tokens) for each example in
  the batch.
- `training` - A flag indicating whether the layer should behave in training mode
  (applying dropout to sparse tensors if applicable) or in inference mode
  (not applying dropout).
  

**Returns**:

  combined features: A tensor containing all the features combined.
- `mask_combined_sequence_sentence` - A binary mask with 1s in place of real
  features in the combined feature tensor, and 0s in padded positions with
  fake features.

## RasaSequenceLayer Objects

```python
class RasaSequenceLayer(RasaCustomLayer)
```

Creates an embedding from all features for a sequence attribute; facilitates MLM.

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

**Arguments**:

- `attribute` - Name of attribute (e.g. `text` or `label`) whose features will be
  processed.
- `attribute_signature` - A dictionary containing two lists of feature signatures,
  one for each feature type (`sentence` or `sequence`) of the given attribute.
- `config` - A model config used for correctly parameterising the underlying layers.
  
  Input shape:
  Tuple of three input tensors:
- `sequence_features` - List of 3-D dense or sparse tensors, each with shape
  `(batch_size, max_seq_length, input_dim)` where `input_dim` can be
  different for sparse vs dense tensors. See the input shape of
  `ConcatenateSparseDenseFeatures` for more information.
- `sentence_features` - List of 3-D dense or sparse tensors, each with shape
  `(batch_size, 1, input_dim)` where `input_dim` can be different for
  sparse vs dense tensors, and can differ from that in
  `sequence_features`. See the input shape of
  `ConcatenateSparseDenseFeatures` for more information.
- `sequence_feature_lengths` - Dense tensor of shape `(batch_size, )`.
  
  Output shape:
- `outputs` - `(batch_size, seq_length, units)` where `units` matches the underlying
  transformer&#x27;s output size (if present), otherwise it matches the output size
  of the `Ffnn` block applied to the combined features, or it&#x27;s the output
  size of the underlying `RasaFeatureCombiningLayer` if the `Ffnn` block has 0
  layers. `seq_length` is the sum of the sequence dimension
  sizes of sequence- and sentence-level features (for details, see the output
  shape of `RasaFeatureCombiningLayer`). If both feature types are present,
  then `seq_length` will be 1 + the length of the longest sequence of real
  tokens across all examples in the given batch.
- `seq_sent_features` - `(batch_size, seq_length, hidden_dim)`, where `hidden_dim` is
  the output size of the underlying `Ffnn` block, or the output size of the
  underlying `RasaFeatureCombiningLayer` if the `Ffnn` block has 0 layers.
- `mask_combined_sequence_sentence` - `(batch_size, seq_length, 1)`
- `token_ids` - `(batch_size, seq_length, id_dim)`. `id_dim` is 2 when no dense
  sequence-level features are present. Otherwise, it&#x27;s arbitrarily chosen to
  match the last dimension size of the first dense sequence-level feature in
  the input list of features.
- `mlm_boolean_mask` - `(batch_size, seq_length, 1)`, empty tensor if not doing MLM.
- `attention_weights` - `(transformer_layers, batch_size, num_transformer_heads,
  seq_length, seq_length)`, empty tensor if the transformer has 0 layers.
  

**Raises**:

  A `TFLayerConfigException` if no feature signatures for sequence-level features
  are provided.
  

**Attributes**:

- `output_units` - The last dimension size of the layer&#x27;s first output (`outputs`).

#### \_\_init\_\_

```python
def __init__(attribute: Text, attribute_signature: Dict[Text, List[FeatureSignature]], config: Dict[Text, Any]) -> None
```

Creates a new `RasaSequenceLayer` object.

#### call

```python
def call(inputs: Tuple[
            List[Union[tf.Tensor, tf.SparseTensor]],
            List[Union[tf.Tensor, tf.SparseTensor]],
            tf.Tensor,
        ], training: bool = False) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
```

Combines all of an attribute&#x27;s features and embeds using a transformer.

**Arguments**:

- `inputs` - Tuple containing:
- `sequence_features` - Dense or sparse tensors representing different
  token-level features.
- `sentence_features` - Dense or sparse tensors representing different
  sentence-level features.
- `sequence_feature_lengths` - A tensor containing the real sequence length
  (the number of real -- not padding -- tokens) for each example in
  the batch.
- `training` - A flag indicating whether the layer should behave in training mode
  (applying dropout to sparse tensors if applicable) or in inference mode
  (not applying dropout).
  

**Returns**:

- `outputs` - Tensor with all features combined, masked (if doing MLM) and
  embedded with a transformer.
- `seq_sent_features` - Tensor with all features combined from just before the
  masking and transformer is applied
- `mask_combined_sequence_sentence` - A binary mask with 1s in place of real
  features in the combined feature tensor, and 0s in padded positions with
  fake features.
- `token_ids` - Tensor with dense token-level features which can serve as
  IDs (unique embeddings) of all the different tokens found in the batch.
  Empty tensor if not doing MLM.
- `mlm_boolean_mask` - A boolean mask with `True` where real tokens in `outputs`
  were masked and `False` elsewhere. Empty tensor if not doing MLM.
- `attention_weights` - Tensor containing self-attention weights received
  from the underlying transformer. Empty tensor if the transformer has 0
  layers.

#### compute\_mask

```python
def compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor
```

Computes binary mask given real sequence lengths.

Takes a 1-D tensor of shape `(batch_size,)` containing the lengths of sequences
(in terms of number of tokens) in the batch. Creates a binary mask of shape
`(batch_size, max_seq_length, 1)` with 1s at positions with real tokens and 0s
elsewhere.

#### prepare\_transformer\_layer

```python
def prepare_transformer_layer(attribute_name: Text, config: Dict[Text, Any], num_layers: int, units: int, drop_rate: float, unidirectional: bool) -> Union[
    TransformerEncoder,
    Callable[
        [tf.Tensor, Optional[tf.Tensor], Optional[Union[tf.Tensor, bool]]],
        Tuple[tf.Tensor, Optional[tf.Tensor]],
    ],
]
```

Creates &amp; returns a transformer encoder, potentially with 0 layers.

