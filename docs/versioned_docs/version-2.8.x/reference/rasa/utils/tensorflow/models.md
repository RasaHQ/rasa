---
sidebar_label: rasa.utils.tensorflow.models
title: rasa.utils.tensorflow.models
---
## RasaModel Objects

```python
class RasaModel(TmpKerasModel)
```

Abstract custom Keras model.

 This model overwrites the following methods:
- train_step
- test_step
- predict_step
- save
- load
Cannot be used as tf.keras.Model.

#### \_\_init\_\_

```python
 | __init__(random_seed: Optional[int] = None, **kwargs: Any) -> None
```

Initialize the RasaModel.

**Arguments**:

- `random_seed` - set the random seed to get reproducible results

#### batch\_loss

```python
 | batch_loss(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> tf.Tensor
```

Calculates the loss for the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The loss of the given batch.

#### prepare\_for\_predict

```python
 | prepare_for_predict() -> None
```

Prepares tf graph fpr prediction.

This method should contain necessary tf calculations
and set self variables that are used in `batch_predict`.
For example, pre calculation of `self.all_labels_embed`.

#### batch\_predict

```python
 | batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

#### train\_step

```python
 | train_step(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, float]
```

Performs a train step using the given batch.

**Arguments**:

- `batch_in` - The batch input.
  

**Returns**:

  Training metrics.

#### test\_step

```python
 | test_step(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, float]
```

Tests the model using the given batch.

This method is used during validation.

**Arguments**:

- `batch_in` - The batch input.
  

**Returns**:

  Testing metrics.

#### predict\_step

```python
 | predict_step(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, tf.Tensor]
```

Predicts the output for the given batch.

**Arguments**:

- `batch_in` - The batch to predict.
  

**Returns**:

  Prediction output.

#### run\_inference

```python
 | run_inference(model_data: RasaModelData, batch_size: Union[int, List[int]] = 1, output_keys_expected: Optional[List[Text]] = None) -> Dict[Text, Union[np.ndarray, Dict[Text, Any]]]
```

Implements bulk inferencing through the model.

**Arguments**:

- `model_data` - Input data to be fed to the model.
- `batch_size` - Size of batches that the generator should create.
- `output_keys_expected` - Keys which are expected in the output.
  The output should be filtered to have only these keys before
  merging it with the output across all batches.
  

**Returns**:

  Model outputs corresponding to the inputs fed.

#### save

```python
 | save(model_file_name: Text, overwrite: bool = True) -> None
```

Save the model to the given file.

**Arguments**:

- `model_file_name` - The file name to save the model to.
- `overwrite` - If &#x27;True&#x27; an already existing model with the same file name will
  be overwritten.

#### load

```python
 | @classmethod
 | load(cls, model_file_name: Text, model_data_example: RasaModelData, predict_data_example: Optional[RasaModelData] = None, finetune_mode: bool = False, *args: Any, **kwargs: Any, *, ,) -> "RasaModel"
```

Loads a model from the given weights.

**Arguments**:

- `model_file_name` - Path to file containing model weights.
- `model_data_example` - Example data point to construct the model architecture.
- `predict_data_example` - Example data point to speed up prediction during
  inference.
- `finetune_mode` - Indicates whether to load the model for further finetuning.
- `*args` - Any other non key-worded arguments.
- `**kwargs` - Any other key-worded arguments.
  

**Returns**:

  Loaded model with weights appropriately set.

#### batch\_to\_model\_data\_format

```python
 | @staticmethod
 | batch_to_model_data_format(batch: Union[Tuple[tf.Tensor], Tuple[np.ndarray]], data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]]) -> Dict[Text, Dict[Text, List[tf.Tensor]]]
```

Convert input batch tensors into batch data format.

Batch contains any number of batch data. The order is equal to the
key-value pairs in session data. As sparse data were converted into (indices,
data, shape) before, this method converts them into sparse tensors. Dense
data is kept.

#### call

```python
 | call(inputs: Union[tf.Tensor, List[tf.Tensor]], training: Optional[tf.Tensor] = None, mask: Optional[tf.Tensor] = None) -> Union[tf.Tensor, List[tf.Tensor]]
```

Calls the model on new inputs.

**Arguments**:

- `inputs` - A tensor or list of tensors.
- `training` - Boolean or boolean scalar tensor, indicating whether to run
  the `Network` in training mode or inference mode.
- `mask` - A mask or list of masks. A mask can be
  either a tensor or None (no mask).
  

**Returns**:

  A tensor if there is a single output, or
  a list of tensors if there are more than one outputs.

## TransformerRasaModel Objects

```python
class TransformerRasaModel(RasaModel)
```

#### adjust\_for\_incremental\_training

```python
 | adjust_for_incremental_training(data_example: Dict[Text, Dict[Text, List[FeatureArray]]], new_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]], old_sparse_feature_sizes: Dict[Text, Dict[Text, List[int]]]) -> None
```

Adjusts the model for incremental training.

First we should check if any of the sparse feature sizes has decreased
and raise an exception if this happens.
If none of them have decreased and any of them has increased, then the
function updates `DenseForSparse` layers, compiles the model, fits a sample
data on it to activate adjusted layer(s) and updates the data signatures.

New and old sparse feature sizes could look like this:
{TEXT: {FEATURE_TYPE_SEQUENCE: [4, 24, 128], FEATURE_TYPE_SENTENCE: [4, 128]}}

**Arguments**:

- `data_example` - a data example that is stored with the ML component.
- `new_sparse_feature_sizes` - sizes of current sparse features.
- `old_sparse_feature_sizes` - sizes of sparse features the model was
  previously trained on.

#### dot\_product\_loss\_layer

```python
 | @property
 | dot_product_loss_layer() -> tf.keras.layers.Layer
```

Returns the dot-product loss layer to use.

**Returns**:

  The loss layer that is used by `_prepare_dot_product_loss`.

#### batch\_loss

```python
 | batch_loss(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> tf.Tensor
```

Calculates the loss for the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The loss of the given batch.

#### batch\_predict

```python
 | batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

