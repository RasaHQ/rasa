---
sidebar_label: rasa.utils.tensorflow.layers
title: rasa.utils.tensorflow.layers
---
## SparseDropout Objects

```python
class SparseDropout(tf.keras.layers.Dropout)
```

Applies Dropout to the input.

Dropout consists in randomly setting
a fraction `rate` of input units to 0 at each update during training time,
which helps prevent overfitting.

**Arguments**:

- `rate` - Fraction of the input units to drop (between 0 and 1).

#### call

```python
 | call(inputs: tf.SparseTensor, training: Optional[Union[tf.Tensor, bool]] = None) -> tf.SparseTensor
```

Apply dropout to sparse inputs.

**Arguments**:

- `inputs` - Input sparse tensor (of any rank).
- `training` - Indicates whether the layer should behave in
  training mode (adding dropout) or in inference mode (doing nothing).
  

**Returns**:

  Output of dropout layer.
  

**Raises**:

  A ValueError if inputs is not a sparse tensor

## DenseForSparse Objects

```python
class DenseForSparse(tf.keras.layers.Dense)
```

Dense layer for sparse input tensor.

Just your regular densely-connected NN layer but for sparse tensors.

`Dense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`).

Note: If the input to the layer has a rank greater than 2, then
it is flattened prior to the initial dot product with `kernel`.

**Arguments**:

- `units` - Positive integer, dimensionality of the output space.
- `activation` - Activation function to use.
  If you don&#x27;t specify anything, no activation is applied
  (ie. &quot;linear&quot; activation: `a(x) = x`).
- `use_bias` - Indicates whether the layer uses a bias vector.
- `kernel_initializer` - Initializer for the `kernel` weights matrix.
- `bias_initializer` - Initializer for the bias vector.
- `reg_lambda` - regularization factor
- `bias_regularizer` - Regularizer function applied to the bias vector.
- `activity_regularizer` - Regularizer function applied to
  the output of the layer (its &quot;activation&quot;)..
- `kernel_constraint` - Constraint function applied to
  the `kernel` weights matrix.
- `bias_constraint` - Constraint function applied to the bias vector.
  
  Input shape:
  N-D tensor with shape: `(batch_size, ..., input_dim)`.
  The most common situation would be
  a 2D input with shape `(batch_size, input_dim)`.
  
  Output shape:
  N-D tensor with shape: `(batch_size, ..., units)`.
  For instance, for a 2D input with shape `(batch_size, input_dim)`,
  the output would have shape `(batch_size, units)`.

#### get\_units

```python
 | get_units() -> int
```

Returns number of output units.

#### get\_kernel

```python
 | get_kernel() -> tf.Tensor
```

Returns kernel tensor.

#### get\_bias

```python
 | get_bias() -> Union[tf.Tensor, None]
```

Returns bias tensor.

#### get\_feature\_type

```python
 | get_feature_type() -> Union[Text, None]
```

Returns a feature type of the data that&#x27;s fed to the layer.

In order to correctly return a feature type, the function heavily relies
on the name of `DenseForSparse` layer to contain the feature type.
Acceptable values of feature types are `FEATURE_TYPE_SENTENCE`
and `FEATURE_TYPE_SEQUENCE`.

**Returns**:

  feature type of dense layer.

#### get\_attribute

```python
 | get_attribute() -> Union[Text, None]
```

Returns the attribute for which this layer was constructed.

For example: TEXT, LABEL, etc.

In order to correctly return an attribute, the function heavily relies
on the name of `DenseForSparse` layer being in the following format:
f&quot;sparse_to_dense.{attribute}_{feature_type}&quot;.

**Returns**:

  attribute of the layer.

#### call

```python
 | call(inputs: tf.SparseTensor) -> tf.Tensor
```

Apply dense layer to sparse inputs.

**Arguments**:

- `inputs` - Input sparse tensor (of any rank).
  

**Returns**:

  Output of dense layer.
  

**Raises**:

  A ValueError if inputs is not a sparse tensor

## RandomlyConnectedDense Objects

```python
class RandomlyConnectedDense(tf.keras.layers.Dense)
```

Layer with dense ouputs that are connected to a random subset of inputs.

`RandomlyConnectedDense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`).
It creates `kernel_mask` to set a fraction of the `kernel` weights to zero.

Note: If the input to the layer has a rank greater than 2, then
it is flattened prior to the initial dot product with `kernel`.

The output is guaranteed to be dense (each output is connected to at least one
input), and no input is disconnected (each input is connected to at least one
output).

At `density = 0.0` the number of trainable weights is `max(input_size, units)`. At
`density = 1.0` this layer is equivalent to `tf.keras.layers.Dense`.

Input shape:
N-D tensor with shape: `(batch_size, ..., input_dim)`.
The most common situation would be
a 2D input with shape `(batch_size, input_dim)`.

Output shape:
N-D tensor with shape: `(batch_size, ..., units)`.
For instance, for a 2D input with shape `(batch_size, input_dim)`,
the output would have shape `(batch_size, units)`.

#### \_\_init\_\_

```python
 | __init__(density: float = 0.2, **kwargs: Any) -> None
```

Declares instance variables with default values.

**Arguments**:

- `density` - Approximate fraction of trainable weights (between 0 and 1).
- `units` - Positive integer, dimensionality of the output space.
- `activation` - Activation function to use.
  If you don&#x27;t specify anything, no activation is applied
  (ie. &quot;linear&quot; activation: `a(x) = x`).
- `use_bias` - Indicates whether the layer uses a bias vector.
- `kernel_initializer` - Initializer for the `kernel` weights matrix.
- `bias_initializer` - Initializer for the bias vector.
- `kernel_regularizer` - Regularizer function applied to
  the `kernel` weights matrix.
- `bias_regularizer` - Regularizer function applied to the bias vector.
- `activity_regularizer` - Regularizer function applied to
  the output of the layer (its &quot;activation&quot;)..
- `kernel_constraint` - Constraint function applied to
  the `kernel` weights matrix.
- `bias_constraint` - Constraint function applied to the bias vector.

#### build

```python
 | build(input_shape: tf.TensorShape) -> None
```

Prepares the kernel mask.

**Arguments**:

- `input_shape` - Shape of the inputs to this layer

#### call

```python
 | call(inputs: tf.Tensor) -> tf.Tensor
```

Processes the given inputs.

**Arguments**:

- `inputs` - What goes into this layer
  

**Returns**:

  The processed inputs.

## Ffnn Objects

```python
class Ffnn(tf.keras.layers.Layer)
```

Feed-forward network layer.

**Arguments**:

- `layer_sizes` - List of integers with dimensionality of the layers.
- `dropout_rate` - Fraction of the input units to drop (between 0 and 1).
- `reg_lambda` - regularization factor.
- `density` - Approximate fraction of trainable weights (between 0 and 1).
- `layer_name_suffix` - Text added to the name of the layers.
  
  Input shape:
  N-D tensor with shape: `(batch_size, ..., input_dim)`.
  The most common situation would be
  a 2D input with shape `(batch_size, input_dim)`.
  
  Output shape:
  N-D tensor with shape: `(batch_size, ..., layer_sizes[-1])`.
  For instance, for a 2D input with shape `(batch_size, input_dim)`,
  the output would have shape `(batch_size, layer_sizes[-1])`.

## Embed Objects

```python
class Embed(tf.keras.layers.Layer)
```

Dense embedding layer.

Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

Output shape:
    N-D tensor with shape: `(batch_size, ..., embed_dim)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, embed_dim)`.

#### \_\_init\_\_

```python
 | __init__(embed_dim: int, reg_lambda: float, layer_name_suffix: Text) -> None
```

Initialize layer.

**Arguments**:

- `embed_dim` - Dimensionality of the output space.
- `reg_lambda` - Regularization factor.
- `layer_name_suffix` - Text added to the name of the layers.

#### call

```python
 | call(x: tf.Tensor) -> tf.Tensor
```

Apply dense layer.

## InputMask Objects

```python
class InputMask(tf.keras.layers.Layer)
```

The layer that masks 15% of the input.

Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

Output shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, input_dim)`.

#### call

```python
 | call(x: tf.Tensor, mask: tf.Tensor, training: Optional[Union[tf.Tensor, bool]] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Randomly mask input sequences.

**Arguments**:

- `x` - Input sequence tensor of rank 3.
- `mask` - A tensor representing sequence mask,
  contains `1` for inputs and `0` for padding.
- `training` - Indicates whether the layer should run in
  training mode (mask inputs) or in inference mode (doing nothing).
  

**Returns**:

  A tuple of masked inputs and boolean mask.

## CRF Objects

```python
class CRF(tf.keras.layers.Layer)
```

CRF layer.

**Arguments**:

- `num_tags` - Positive integer, number of tags.
- `reg_lambda` - regularization factor.
- `name` - Optional name of the layer.

#### call

```python
 | call(logits: tf.Tensor, sequence_lengths: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]
```

Decodes the highest scoring sequence of tags.

**Arguments**:

- `logits` - A [batch_size, max_seq_len, num_tags] tensor of
  unary potentials.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
  

**Returns**:

  A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
  Contains the highest scoring tag indices.
  A [batch_size, max_seq_len] matrix, with dtype `tf.float32`.
  Contains the confidence values of the highest scoring tag indices.

#### loss

```python
 | loss(logits: tf.Tensor, tag_indices: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor
```

Computes the log-likelihood of tag sequences in a CRF.

**Arguments**:

- `logits` - A [batch_size, max_seq_len, num_tags] tensor of unary potentials
  to use as input to the CRF layer.
- `tag_indices` - A [batch_size, max_seq_len] matrix of tag indices for which
  we compute the log-likelihood.
- `sequence_lengths` - A [batch_size] vector of true sequence lengths.
  

**Returns**:

  Negative mean log-likelihood of all examples,
  given the sequence of tag indices.

#### f1\_score

```python
 | f1_score(tag_ids: tf.Tensor, pred_ids: tf.Tensor, mask: tf.Tensor) -> tf.Tensor
```

Calculates f1 score for train predictions

## DotProductLoss Objects

```python
class DotProductLoss(tf.keras.layers.Layer)
```

Abstract dot-product loss layer class.

Idea based on StarSpace paper: http://arxiv.org/abs/1709.03856

Implements similarity methods
* `sim` (computes a similarity between vectors)
* `get_similarities_and_confidences_from_embeddings` (calls `sim` and also computes
    confidence values)

Specific loss functions (single- or multi-label) must be implemented in child
classes.

#### \_\_init\_\_

```python
 | __init__(num_candidates: int, scale_loss: bool = False, constrain_similarities: bool = True, model_confidence: Text = SOFTMAX, similarity_type: Text = INNER, name: Optional[Text] = None, **kwargs: Any, ,)
```

Declares instance variables with default values.

**Arguments**:

- `num_candidates` - Number of labels besides the positive one. Depending on
  whether single- or multi-label loss is implemented (done in
  sub-classes), these can be all negative example labels, or a mixture of
  negative and further positive labels, respectively.
- `scale_loss` - Boolean, if `True` scale loss inverse proportionally to
  the confidence of the correct prediction.
- `constrain_similarities` - Boolean, if `True` applies sigmoid on all
  similarity terms and adds to the loss function to
  ensure that similarity values are approximately bounded.
  Used inside _loss_cross_entropy() only.
- `model_confidence` - Normalization of confidence values during inference.
  Possible values are `SOFTMAX` and `LINEAR_NORM`.
- `similarity_type` - Similarity measure to use, either `cosine` or `inner`.
- `name` - Optional name of the layer.
  

**Raises**:

- `TFLayerConfigException` - When `similarity_type` is not one of `COSINE` or
  `INNER`.

#### sim

```python
 | sim(a: tf.Tensor, b: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor
```

Calculates similarity between `a` and `b`.

Operates on the last dimension. When `a` and `b` are vectors, then `sim`
computes either the dot-product, or the cosine of the angle between `a` and `b`,
depending on `self.similarity_type`.
Specifically, when the similarity type is `INNER`, then we compute the scalar
product `a . b`. When the similarity type is `COSINE`, we compute
`a . b / (|a| |b|)`, i.e. the cosine of the angle between `a` and `b`.

**Arguments**:

- `a` - Any float tensor
- `b` - Any tensor of the same shape and type as `a`
- `mask` - Mask (should contain 1s for inputs and 0s for padding). Note, that
  `len(mask.shape) == len(a.shape) - 1` should hold.
  

**Returns**:

  Similarities between vectors in `a` and `b`.

#### get\_similarities\_and\_confidences\_from\_embeddings

```python
 | get_similarities_and_confidences_from_embeddings(input_embeddings: tf.Tensor, label_embeddings: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Computes similary between input and label embeddings and model&#x27;s confidence.

First compute the similarity from embeddings and then apply an activation
function if needed to get the confidence.

**Arguments**:

- `input_embeddings` - Embeddings of input.
- `label_embeddings` - Embeddings of labels.
- `mask` - Mask (should contain 1s for inputs and 0s for padding). Note, that
  `len(mask.shape) == len(a.shape) - 1` should hold.
  

**Returns**:

  similarity between input and label embeddings and model&#x27;s prediction
  confidence for each label.

#### call

```python
 | call(*args: Any, **kwargs: Any) -> Tuple[tf.Tensor, tf.Tensor]
```

Layer&#x27;s logic - to be implemented in child class.

#### apply\_mask\_and\_scaling

```python
 | apply_mask_and_scaling(loss: tf.Tensor, mask: Optional[tf.Tensor]) -> tf.Tensor
```

Scales the loss and applies the mask if necessary.

**Arguments**:

- `loss` - The loss tensor
- `mask` - (Optional) A mask to multiply with the loss
  

**Returns**:

  The scaled loss, potentially averaged over the sequence
  dimension.

## SingleLabelDotProductLoss Objects

```python
class SingleLabelDotProductLoss(DotProductLoss)
```

Single-label dot-product loss layer.

This loss layer assumes that only one output (label) is correct for any given input.

#### \_\_init\_\_

```python
 | __init__(num_candidates: int, scale_loss: bool = False, constrain_similarities: bool = True, model_confidence: Text = SOFTMAX, similarity_type: Text = INNER, name: Optional[Text] = None, loss_type: Text = CROSS_ENTROPY, mu_pos: float = 0.8, mu_neg: float = -0.2, use_max_sim_neg: bool = True, neg_lambda: float = 0.5, same_sampling: bool = False, **kwargs: Any, ,) -> None
```

Declares instance variables with default values.

**Arguments**:

- `num_candidates` - Positive integer, the number of incorrect labels;
  the algorithm will minimize their similarity to the input.
- `loss_type` - The type of the loss function, either `cross_entropy` or
  `margin`.
- `mu_pos` - Indicates how similar the algorithm should
  try to make embedding vectors for correct labels;
  should be 0.0 &lt; ... &lt; 1.0 for `cosine` similarity type.
- `mu_neg` - Maximum negative similarity for incorrect labels,
  should be -1.0 &lt; ... &lt; 1.0 for `cosine` similarity type.
- `use_max_sim_neg` - If `True` the algorithm only minimizes
  maximum similarity over incorrect intent labels,
  used only if `loss_type` is set to `margin`.
- `neg_lambda` - The scale of how important it is to minimize
  the maximum similarity between embeddings of different labels,
  used only if `loss_type` is set to `margin`.
- `scale_loss` - If `True` scale loss inverse proportionally to
  the confidence of the correct prediction.
- `similarity_type` - Similarity measure to use, either `cosine` or `inner`.
- `name` - Optional name of the layer.
- `same_sampling` - If `True` sample same negative labels
  for the whole batch.
- `constrain_similarities` - If `True` and loss_type is `cross_entropy`, a
  sigmoid loss term is added to the total loss to ensure that similarity
  values are approximately bounded.
- `model_confidence` - Normalization of confidence values during inference.
  Possible values are `SOFTMAX` and `LINEAR_NORM`.

#### call

```python
 | call(inputs_embed: tf.Tensor, labels_embed: tf.Tensor, labels: tf.Tensor, all_labels_embed: tf.Tensor, all_labels: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Calculate loss and accuracy.

**Arguments**:

- `inputs_embed` - Embedding tensor for the batch inputs;
  shape `(batch_size, ..., num_features)`
- `labels_embed` - Embedding tensor for the batch labels;
  shape `(batch_size, ..., num_features)`
- `labels` - Tensor representing batch labels; shape `(batch_size, ..., 1)`
- `all_labels_embed` - Embedding tensor for the all labels;
  shape `(num_labels, num_features)`
- `all_labels` - Tensor representing all labels; shape `(num_labels, 1)`
- `mask` - Optional mask, contains `1` for inputs and `0` for padding;
  shape `(batch_size, 1)`
  

**Returns**:

- `loss` - Total loss.
- `accuracy` - Training accuracy.

## MultiLabelDotProductLoss Objects

```python
class MultiLabelDotProductLoss(DotProductLoss)
```

Multi-label dot-product loss layer.

This loss layer assumes that multiple outputs (labels) can be correct for any given
input. To accomodate for this, we use a sigmoid cross-entropy loss here.

#### \_\_init\_\_

```python
 | __init__(num_candidates: int, scale_loss: bool = False, constrain_similarities: bool = True, model_confidence: Text = SOFTMAX, similarity_type: Text = INNER, name: Optional[Text] = None, **kwargs: Any, ,) -> None
```

Declares instance variables with default values.

**Arguments**:

- `num_candidates` - Positive integer, the number of candidate labels.
- `scale_loss` - If `True` scale loss inverse proportionally to
  the confidence of the correct prediction.
- `similarity_type` - Similarity measure to use, either `cosine` or `inner`.
- `name` - Optional name of the layer.
- `constrain_similarities` - Boolean, if `True` applies sigmoid on all
  similarity terms and adds to the loss function to
  ensure that similarity values are approximately bounded.
  Used inside _loss_cross_entropy() only.
- `model_confidence` - Normalization of confidence values during inference.
  Possible values are `SOFTMAX` and `LINEAR_NORM`.

#### call

```python
 | call(batch_inputs_embed: tf.Tensor, batch_labels_embed: tf.Tensor, batch_labels_ids: tf.Tensor, all_labels_embed: tf.Tensor, all_labels_ids: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Calculates loss and accuracy.

**Arguments**:

- `batch_inputs_embed` - Embeddings of the batch inputs (e.g. featurized
  trackers); shape `(batch_size, 1, num_features)`
- `batch_labels_embed` - Embeddings of the batch labels (e.g. featurized intents
  for IntentTED);
  shape `(batch_size, max_num_labels_per_input, num_features)`
- `batch_labels_ids` - Batch label indices (e.g. indices of the intents). We
  assume that indices are integers that run from `0` to
  `(number of labels) - 1`.
  shape `(batch_size, max_num_labels_per_input, 1)`
- `all_labels_embed` - Embeddings for all labels in the domain;
  shape `(batch_size, num_features)`
- `all_labels_ids` - Indices for all labels in the domain;
  shape `(num_labels, 1)`
- `mask` - Optional sequence mask, which contains `1` for inputs and `0` for
  padding.
  

**Returns**:

- `loss` - Total loss (based on StarSpace http://arxiv.org/abs/1709.03856);
  scalar
- `accuracy` - Training accuracy; scalar

