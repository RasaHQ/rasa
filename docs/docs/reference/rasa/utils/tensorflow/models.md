---
sidebar_label: models
title: rasa.utils.tensorflow.models
---

## RasaModel Objects

```python
class RasaModel(tf.keras.models.Model)
```

Completely override all public methods of keras Model.

Cannot be used as tf.keras.Model

#### \_\_init\_\_

```python
 | __init__(random_seed: Optional[int] = None, tensorboard_log_dir: Optional[Text] = None, tensorboard_log_level: Optional[Text] = "epoch", checkpoint_model: Optional[bool] = False, **kwargs, ,) -> None
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
 | batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, tf.Tensor]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

#### fit

```python
 | fit(model_data: RasaModelData, epochs: int, batch_size: Union[List[int], int], evaluate_on_num_examples: int, evaluate_every_num_epochs: int, batch_strategy: Text, silent: bool = False, loading: bool = False, eager: bool = False) -> None
```

Fit model data.

#### train\_on\_batch

```python
 | train_on_batch(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> None
```

Train on batch.

#### load

```python
 | @classmethod
 | load(cls, model_file_name: Text, model_data_example: RasaModelData, finetune_mode: bool = False, *args, **kwargs, *, ,) -> "RasaModel"
```

Loads a model from the given weights.

**Arguments**:

- `model_file_name` - Path to file containing model weights.
- `model_data_example` - Example data point to construct the model architecture.
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

#### linearly\_increasing\_batch\_size

```python
 | @staticmethod
 | linearly_increasing_batch_size(epoch: int, batch_size: Union[List[int], int], epochs: int) -> int
```

Linearly increase batch size with every epoch.

The idea comes from https://arxiv.org/abs/1711.00489.

## TransformerRasaModel Objects

```python
class TransformerRasaModel(RasaModel)
```

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
 | batch_predict(batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]) -> Dict[Text, tf.Tensor]
```

Predicts the output of the given batch.

**Arguments**:

- `batch_in` - The batch.
  

**Returns**:

  The output to predict.

