---
sidebar_label: rasa.utils.tensorflow.callback
title: rasa.utils.tensorflow.callback
---
## RasaTrainingLogger Objects

```python
class RasaTrainingLogger(tf.keras.callbacks.Callback)
```

Callback for logging the status of training.

#### \_\_init\_\_

```python
def __init__(epochs: int, silent: bool) -> None
```

Initializes the callback.

**Arguments**:

- `epochs` - Total number of epochs.
- `silent` - If &#x27;True&#x27; the entire progressbar wrapper is disabled.

#### on\_epoch\_end

```python
def on_epoch_end(epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None
```

Updates the logging output on every epoch end.

**Arguments**:

- `epoch` - The current epoch.
- `logs` - The training metrics.

#### on\_train\_end

```python
def on_train_end(logs: Optional[Dict[Text, Any]] = None) -> None
```

Closes the progress bar after training.

**Arguments**:

- `logs` - The training metrics.

## RasaModelCheckpoint Objects

```python
class RasaModelCheckpoint(tf.keras.callbacks.Callback)
```

Callback for saving intermediate model checkpoints.

#### \_\_init\_\_

```python
def __init__(checkpoint_dir: Path) -> None
```

Initializes the callback.

**Arguments**:

- `checkpoint_dir` - Directory to store checkpoints to.

#### on\_epoch\_end

```python
def on_epoch_end(epoch: int, logs: Optional[Dict[Text, Any]] = None) -> None
```

Save the model on epoch end if the model has improved.

**Arguments**:

- `epoch` - The current epoch.
- `logs` - The training metrics.

