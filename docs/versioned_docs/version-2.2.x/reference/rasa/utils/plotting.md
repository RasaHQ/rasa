---
sidebar_label: plotting
title: rasa.utils.plotting
---

#### plot\_confusion\_matrix

```python
plot_confusion_matrix(confusion_matrix: np.ndarray, classes: Union[np.ndarray, List[Text]], normalize: bool = False, title: Text = "Confusion matrix", color_map: Any = None, zmin: int = 1, output_file: Optional[Text] = None) -> None
```

Print and plot the provided confusion matrix.
Normalization can be applied by setting `normalize=True`.

**Arguments**:

- `confusion_matrix` - confusion matrix to plot
- `classes` - class labels
- `normalize` - If set to true, normalization will be applied.
- `title` - title of the plot
- `color_map` - color mapping
  zmin:
- `output_file` - output file to save plot to

#### plot\_histogram

```python
plot_histogram(hist_data: List[List[float]], title: Text, output_file: Optional[Text] = None) -> None
```

Plot a side-by-side comparative histogram of the confidence distribution (misses and hits).

**Arguments**:

- `hist_data` - histogram data
- `title` - title of the plot
- `output_file` - output file to save the plot to

#### plot\_curve

```python
plot_curve(output_directory: Text, number_of_examples: List[int], x_label_text: Text, y_label_text: Text, graph_path: Text) -> None
```

Plot the results from a model comparison.

**Arguments**:

- `output_directory` - Output directory to save resulting plots to
- `number_of_examples` - Number of examples per run
- `x_label_text` - text for the x axis
- `y_label_text` - text for the y axis
- `graph_path` - output path of the plot

