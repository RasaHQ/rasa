import logging
import itertools
import os
from functools import wraps

import numpy as np
from typing import Any, Callable, List, Optional, Text, TypeVar, Union, Tuple
import matplotlib
from matplotlib.ticker import FormatStrFormatter

import rasa.shared.utils.io
from rasa.constants import RESULTS_FILE

logger = logging.getLogger(__name__)


def _fix_matplotlib_backend() -> None:
    """Tries to fix a broken matplotlib backend."""
    try:
        backend = matplotlib.get_backend()
    except Exception:  # skipcq:PYL-W0703
        logger.error(
            "Cannot retrieve Matplotlib backend, likely due to a compatibility "
            "issue with system dependencies. Please refer to the documentation: "
            "https://matplotlib.org/stable/tutorials/introductory/usage.html#backends"
        )
        raise

    # At first, matplotlib will be initialized with default OS-specific
    # available backend
    if backend == "TkAgg":
        try:
            # on OSX sometimes the tkinter package is broken and can't be imported.
            # we'll try to import it and if it fails we will use a different backend
            import tkinter  # noqa: 401
        except (ImportError, ModuleNotFoundError):
            logger.debug("Setting matplotlib backend to 'agg'")
            matplotlib.use("agg")

    # if no backend is set by default, we'll try to set it up manually
    elif backend is None:  # pragma: no cover
        try:
            # If the `tkinter` package is available, we can use the `TkAgg` backend
            import tkinter  # noqa: 401

            logger.debug("Setting matplotlib backend to 'TkAgg'")
            matplotlib.use("TkAgg")
        except (ImportError, ModuleNotFoundError):
            logger.debug("Setting matplotlib backend to 'agg'")
            matplotlib.use("agg")


ReturnType = TypeVar("ReturnType")
FuncType = Callable[..., ReturnType]
_MATPLOTLIB_BACKEND_FIXED = False


def _needs_matplotlib_backend(func: FuncType) -> FuncType:
    """Decorator to fix matplotlib backend before calling a function."""

    @wraps(func)
    def inner(*args: Any, **kwargs: Any) -> ReturnType:  # type: ignore
        """Replacement function that fixes matplotlib backend."""
        global _MATPLOTLIB_BACKEND_FIXED
        if not _MATPLOTLIB_BACKEND_FIXED:
            _fix_matplotlib_backend()
            _MATPLOTLIB_BACKEND_FIXED = True
        return func(*args, **kwargs)

    return inner


@_needs_matplotlib_backend
def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    classes: Union[np.ndarray, List[Text]],
    normalize: bool = False,
    title: Text = "Confusion matrix",
    color_map: Any = None,
    zmin: int = 1,
    output_file: Optional[Text] = None,
) -> None:
    """
    Print and plot the provided confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        confusion_matrix: confusion matrix to plot
        classes: class labels
        normalize: If set to true, normalization will be applied.
        title: title of the plot
        color_map: color mapping
        zmin:
        output_file: output file to save plot to

    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    zmax = confusion_matrix.max() if len(confusion_matrix) > 0 else 1
    plt.clf()
    if not color_map:
        color_map = plt.cm.Blues
    plt.imshow(
        confusion_matrix,
        interpolation="nearest",
        cmap=color_map,
        aspect="auto",
        norm=LogNorm(vmin=zmin, vmax=zmax),
    )
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        confusion_matrix = (
            confusion_matrix.astype("float")
            / confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        logger.info(f"Normalized confusion matrix: \n{confusion_matrix}")
    else:
        logger.info(f"Confusion matrix, without normalization: \n{confusion_matrix}")

    thresh = zmax / 2.0
    for i, j in itertools.product(
        range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            confusion_matrix[i, j],
            horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # save confusion matrix to file before showing it
    if output_file:
        fig = plt.gcf()
        fig.set_size_inches(20, 20)
        fig.savefig(output_file, bbox_inches="tight")


def _extract_paired_histogram_specification(
    histogram_data: List[List[float]],
    num_bins: int,
    density: bool,
    x_pad_fraction: float,
    y_pad_fraction: float,
) -> Tuple[List[float], List[List[float]], List[float], Tuple[float, float]]:
    """Extracts all information from the data needed to plot a paired histogram.

    Args:
        histogram_data: Two data vectors
        num_bins: Number of bins to be used for the histogram
        density: If true, generate information for a probability density histogram
        x_pad_fraction: Percentage of extra space in the horizontal direction
        y_pad_fraction: Percentage of extra space in the vertical direction

    Returns:
        The bins, values, ranges of either x-axis, and the range of the y-axis

    Raises:
        ValueError: If histogram_data does not contain values.
    """
    if not histogram_data or not np.concatenate(histogram_data).size:
        rasa.shared.utils.io.raise_warning("No data to plot paired histogram.")
        raise ValueError("No data to plot paired histogram.")
    min_data_value: float = np.min(np.concatenate(histogram_data))
    max_data_value: float = np.max(np.concatenate(histogram_data))
    bin_width = (max_data_value - min_data_value) / num_bins
    bins = [
        min_data_value + i * bin_width
        # `bins` describes the _boundaries_ of the bins, so we need
        # 2 extra - one at the beginning and one at the end
        for i in range(num_bins + 2)
    ]
    histograms = [
        # A list of counts - how often a value in `data` falls into a particular bin
        list(np.histogram(data, bins=bins, density=density)[0])
        for data in histogram_data
    ]

    y_padding = 0.5 * bin_width + y_pad_fraction * bin_width

    if density:
        # Get the maximum count across both histograms, and scale it
        # with `x_pad_fraction`
        v = max([(1.0 + x_pad_fraction) * max(histogram) for histogram in histograms])
        # When we plot the PDF, let both x-axes run to the same value
        # so it's easier to compare visually
        x_ranges = [v, v]
    else:
        # For the left and right histograms, get the largest counts and scale them
        # by `x_pad_fraction` to get the maximum x-values displayed
        x_ranges = [(1.0 + x_pad_fraction) * max(histogram) for histogram in histograms]

    try:
        bin_of_first_non_zero_tally = min(
            [[bool(v) for v in histogram].index(True) for histogram in histograms]
        )
    except ValueError:
        bin_of_first_non_zero_tally = 0

    y_range = (
        # Start plotting where the data starts (ignore empty bins at the low end)
        bins[bin_of_first_non_zero_tally] - y_padding,
        # The y_padding adds half a bin width, as we want the bars to be
        # _centered_ on the bins. We take the next-to-last element of `bins`,
        # because that is the beginning of the last bin.
        bins[-2] + y_padding,
    )

    return bins, histograms, x_ranges, y_range


@_needs_matplotlib_backend
def plot_paired_histogram(
    histogram_data: List[List[float]],
    title: Text,
    output_file: Optional[Text] = None,
    num_bins: int = 25,
    colors: Tuple[Text, Text] = ("#009292", "#920000"),  # (dark cyan, dark red)
    axes_label: Tuple[Text, Text] = ("Correct", "Wrong"),
    frame_label: Tuple[Text, Text] = ("Number of Samples", "Confidence"),
    density: bool = False,
    x_pad_fraction: float = 0.05,
    y_pad_fraction: float = 0.10,
) -> None:
    """Plots a side-by-side comparative histogram of the confidence distribution.

    Args:
        histogram_data: Two data vectors
        title: Title to be displayed above the plot
        output_file: File to save the plot to
        num_bins: Number of bins to be used for the histogram
        colors: Left and right bar colors as hex color strings
        axes_label: Labels shown above the left and right histogram,
            respectively
        frame_label: Labels shown below and on the left of the
            histogram, respectively
        density: If true, generate a probability density histogram
        x_pad_fraction: Percentage of extra space in the horizontal direction
        y_pad_fraction: Percentage of extra space in the vertical direction
    """
    if num_bins <= 2:
        rasa.shared.utils.io.raise_warning(
            f"Number {num_bins} of paired histogram bins must be at least 3."
        )
        return

    try:
        bins, tallies, x_ranges, y_range = _extract_paired_histogram_specification(
            histogram_data,
            num_bins,
            density=density,
            x_pad_fraction=x_pad_fraction,
            y_pad_fraction=y_pad_fraction,
        )
    except (ValueError, TypeError) as e:
        rasa.shared.utils.io.raise_warning(
            f"Unable to plot paired histogram '{title}': {e}"
        )
        return
    yticks = [float(f"{x:.2f}") for x in bins]

    import matplotlib.pyplot as plt

    plt.gcf().clear()

    fig, axes = plt.subplots(ncols=2, sharey=True)
    for side in range(2):
        axes[side].barh(
            bins[:-1],
            tallies[side],
            height=np.diff(bins),
            align="center",
            color=colors[side],
            linewidth=1,
            edgecolor="white",
        )
        axes[side].set(title=axes_label[side])
        axes[side].set(yticks=yticks, xlim=(0, x_ranges[side]), ylim=y_range)

    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

    axes[0].invert_xaxis()
    axes[0].yaxis.tick_right()

    # Add the title
    fig.suptitle(title, fontsize="x-large", fontweight="bold")

    # Add hidden plot to correctly add x and y labels (frame_label)
    fig.add_subplot(111, frameon=False)

    # Hide tick and tick label of the unused axis
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.xlabel(frame_label[0])
    plt.ylabel(frame_label[1])

    if output_file:
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.tight_layout(w_pad=0)
        fig.savefig(output_file, bbox_inches="tight")


@_needs_matplotlib_backend
def plot_curve(
    output_directory: Text,
    number_of_examples: List[int],
    x_label_text: Text,
    y_label_text: Text,
    graph_path: Text,
) -> None:
    """Plot the results from a model comparison.

    Args:
        output_directory: Output directory to save resulting plots to
        number_of_examples: Number of examples per run
        x_label_text: text for the x axis
        y_label_text: text for the y axis
        graph_path: output path of the plot
    """
    import matplotlib.pyplot as plt

    plt.gcf().clear()

    ax = plt.gca()

    # load results from file
    data = rasa.shared.utils.io.read_json_file(
        os.path.join(output_directory, RESULTS_FILE)
    )
    x = number_of_examples

    # compute mean of all the runs for different configs
    for label in data.keys():
        if len(data[label]) == 0:
            continue
        mean = np.mean(data[label], axis=0)
        std = np.std(data[label], axis=0)
        ax.plot(x, mean, label=label, marker=".")
        ax.fill_between(
            x,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            color="#6b2def",
            alpha=0.2,
        )
    ax.legend(loc=4)

    ax.set_xlabel(x_label_text)
    ax.set_ylabel(y_label_text)

    plt.savefig(graph_path, format="pdf")

    logger.info(f"Comparison graph saved to '{graph_path}'.")
