---
sidebar_label: rasa.utils.plotting
title: rasa.utils.plotting
---
#### plot\_confusion\_matrix

```python
@_needs_matplotlib_backend
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

#### plot\_paired\_histogram

```python
@_needs_matplotlib_backend
plot_paired_histogram(histogram_data: List[List[float]], title: Text, output_file: Optional[Text] = None, num_bins: int = 25, colors: Tuple[Text, Text] = ("#009292", "#920000"), axes_label: Tuple[Text, Text] = ("Correct", "Wrong"), frame_label: Tuple[Text, Text] = ("Number of Samples", "Confidence"), density: bool = False, x_pad_fraction: float = 0.05, y_pad_fraction: float = 0.10) -> None
```

Plots a side-by-side comparative histogram of the confidence distribution.

**Arguments**:

- `histogram_data` - Two data vectors
- `title` - Title to be displayed above the plot
- `output_file` - File to save the plot to
- `num_bins` - Number of bins to be used for the histogram
- `colors` - Left and right bar colors as hex color strings
- `axes_label` - Labels shown above the left and right histogram,
  respectively
- `frame_label` - Labels shown below and on the left of the
  histogram, respectively
- `density` - If true, generate a probability density histogram
- `x_pad_fraction` - Percentage of extra space in the horizontal direction
- `y_pad_fraction` - Percentage of extra space in the vertical direction

#### plot\_curve

```python
@_needs_matplotlib_backend
plot_curve(output_directory: Text, number_of_examples: List[int], x_label_text: Text, y_label_text: Text, graph_path: Text) -> None
```

Plot the results from a model comparison.

**Arguments**:

- `output_directory` - Output directory to save resulting plots to
- `number_of_examples` - Number of examples per run
- `x_label_text` - text for the x axis
- `y_label_text` - text for the y axis
- `graph_path` - output path of the plot

