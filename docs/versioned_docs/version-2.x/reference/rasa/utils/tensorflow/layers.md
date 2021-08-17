---
sidebar_label: layers
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

- `rate` - Float between 0 and 1; fraction of the input units to drop.

#### call

```python
 | call(inputs: tf.SparseTensor, training: Optional[Union[tf.Tensor, bool]] = None) -> tf.SparseTensor
```

Apply dropout to sparse inputs.

**Arguments**:

- `inputs` - Input sparse tensor (of any rank).
- `training` - Python boolean indicating whether the layer should behave in
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
- `use_bias` - Boolean, whether the layer uses a bias vector.
- `kernel_initializer` - Initializer for the `kernel` weights matrix.
- `bias_initializer` - Initializer for the bias vector.
- `reg_lambda` - Float, regularization factor.
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

## DenseWithSparseWeights Objects

```python
class DenseWithSparseWeights(tf.keras.layers.Dense)
```

Just your regular densely-connected NN layer but with sparse weights.

`Dense` implements the operation:
`output = activation(dot(input, kernel) + bias)`
where `activation` is the element-wise activation function
passed as the `activation` argument, `kernel` is a weights matrix
created by the layer, and `bias` is a bias vector created by the layer
(only applicable if `use_bias` is `True`).
It creates `kernel_mask` to set fraction of the `kernel` weights to zero.

Note: If the input to the layer has a rank greater than 2, then
it is flattened prior to the initial dot product with `kernel`.

**Arguments**:

- `sparsity` - Float between 0 and 1. Fraction of the `kernel`
  weights to set to zero.
- `units` - Positive integer, dimensionality of the output space.
- `activation` - Activation function to use.
  If you don&#x27;t specify anything, no activation is applied
  (ie. &quot;linear&quot; activation: `a(x) = x`).
- `use_bias` - Boolean, whether the layer uses a bias vector.
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
  
  Input shape:
  N-D tensor with shape: `(batch_size, ..., input_dim)`.
  The most common situation would be
  a 2D input with shape `(batch_size, input_dim)`.
  
  Output shape:
  N-D tensor with shape: `(batch_size, ..., units)`.
  For instance, for a 2D input with shape `(batch_size, input_dim)`,
  the output would have shape `(batch_size, units)`.

## Ffnn Objects

```python
class Ffnn(tf.keras.layers.Layer)
```

Feed-forward network layer.

**Arguments**:

- `layer_sizes` - List of integers with dimensionality of the layers.
- `dropout_rate` - Float between 0 and 1; fraction of the input units to drop.
- `reg_lambda` - Float, regularization factor.
- `sparsity` - Float between 0 and 1. Fraction of the `kernel`
  weights to set to zero.
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
- `training` - Python boolean indicating whether the layer should behave in
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
- `reg_lambda` - Float; regularization factor.
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

Dot-product loss layer.

#### \_\_init\_\_

```python
 | __init__(num_neg: int, loss_type: Text, mu_pos: float, mu_neg: float, use_max_sim_neg: bool, neg_lambda: float, scale_loss: bool, similarity_type: Text, name: Optional[Text] = None, same_sampling: bool = False, constrain_similarities: bool = True, model_confidence: Text = SOFTMAX) -> None
```

Declare instance variables with default values.

**Arguments**:

- `num_neg` - Positive integer, the number of incorrect labels;
  the algorithm will minimize their similarity to the input.
- `loss_type` - The type of the loss function, either &#x27;cross_entropy&#x27; or &#x27;margin&#x27;.
- `mu_pos` - Float, indicates how similar the algorithm should
  try to make embedding vectors for correct labels;
  should be 0.0 &lt; ... &lt; 1.0 for &#x27;cosine&#x27; similarity type.
- `mu_neg` - Float, maximum negative similarity for incorrect labels,
  should be -1.0 &lt; ... &lt; 1.0 for &#x27;cosine&#x27; similarity type.
- `use_max_sim_neg` - Boolean, if &#x27;True&#x27; the algorithm only minimizes
  maximum similarity over incorrect intent labels,
  used only if &#x27;loss_type&#x27; is set to &#x27;margin&#x27;.
- `neg_lambda` - Float, the scale of how important is to minimize
  the maximum similarity between embeddings of different labels,
  used only if &#x27;loss_type&#x27; is set to &#x27;margin&#x27;.
- `scale_loss` - Boolean, if &#x27;True&#x27; scale loss inverse proportionally to
  the confidence of the correct prediction.
- `similarity_type` - Similarity measure to use, either &#x27;cosine&#x27; or &#x27;inner&#x27;.
- `name` - Optional name of the layer.
- `same_sampling` - Boolean, if &#x27;True&#x27; sample same negative labels
  for the whole batch.
- `constrain_similarities` - Boolean, if &#x27;True&#x27; applies sigmoid on all
  similarity terms and adds to the loss function to
  ensure that similarity values are approximately bounded.
  Used inside _loss_cross_entropy() only.
- `model_confidence` - Model confidence to be returned during inference.
  Possible values - &#x27;softmax&#x27; and &#x27;linear_norm&#x27;.
  

**Raises**:

- `LayerConfigException` - When `similarity_type` is not one of &#x27;cosine&#x27; or &#x27;inner&#x27;.

#### sim

```python
 | sim(a: tf.Tensor, b: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor
```

Calculate similarity between given tensors.

#### similarity\_confidence\_from\_embeddings

```python
 | similarity_confidence_from_embeddings(input_embeddings: tf.Tensor, label_embeddings: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Computes similarity between input and label embeddings and model&#x27;s confidence.

First compute the similarity from embeddings and then apply an activation
function if needed to get the confidence.

**Arguments**:

- `input_embeddings` - Embeddings of input.
- `label_embeddings` - Embeddings of labels.
- `mask` - Mask over input and output sequence.
  

**Returns**:

  similarity between input and label embeddings and model&#x27;s prediction confidence for each label.

#### call

```python
 | call(inputs_embed: tf.Tensor, labels_embed: tf.Tensor, labels: tf.Tensor, all_labels_embed: tf.Tensor, all_labels: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Tuple[tf.Tensor, tf.Tensor]
```

Calculate loss and accuracy.

**Arguments**:

- `inputs_embed` - Embedding tensor for the batch inputs.
- `labels_embed` - Embedding tensor for the batch labels.
- `labels` - Tensor representing batch labels.
- `all_labels_embed` - Embedding tensor for the all labels.
- `all_labels` - Tensor representing all labels.
- `mask` - Optional tensor representing sequence mask,
  contains `1` for inputs and `0` for padding.
  

**Returns**:

- `loss` - Total loss.
- `accuracy` - Training accuracy.

