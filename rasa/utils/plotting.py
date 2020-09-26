import logging
import itertools
import os

import numpy as np
from typing import List, Text, Optional, Union, Any
import matplotlib

import rasa.shared.utils.io
from rasa.constants import RESULTS_FILE

logger = logging.getLogger(__name__)


def _fix_matplotlib_backend() -> None:
    """Tries to fix a broken matplotlib backend..."""
    # At first, matplotlib will be initialized with default OS-specific
    # available backend
    if matplotlib.get_backend() == "TkAgg":
        try:
            # on OSX sometimes the tkinter package is broken and can't be imported.
            # we'll try to import it and if it fails we will use a different backend
            import tkinter  # skipcq: PYL-W0611
        except (ImportError, ModuleNotFoundError):
            logger.debug("Setting matplotlib backend to 'agg'")
            matplotlib.use("agg")

    # if no backend is set by default, we'll try to set it up manually
    elif matplotlib.get_backend() is None:  # pragma: no cover
        try:
            # If the `tkinter` package is available, we can use the `TkAgg` backend
            import tkinter  # skipcq: PYL-W0611

            logger.debug("Setting matplotlib backend to 'TkAgg'")
            matplotlib.use("TkAgg")
        except (ImportError, ModuleNotFoundError):
            logger.debug("Setting matplotlib backend to 'agg'")
            matplotlib.use("agg")


# we call the fix as soon as this package gets imported
_fix_matplotlib_backend()


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


def plot_histogram(
    hist_data: List[List[float]], title: Text, output_file: Optional[Text] = None
) -> None:
    """
    Plot a histogram of the confidence distribution of the predictions in two columns.

    Args:
        hist_data: histogram data
        output_file: output file to save the plot ot
    """
    import matplotlib.pyplot as plt

    plt.gcf().clear()

    # Wine-ish colour for the confidences of hits.
    # Blue-ish colour for the confidences of misses.
    colors = ["#009292", "#920000"]
    bins = [0.05 * i for i in range(1, 21)]

    plt.xlim([0, 1])
    plt.hist(hist_data, bins=bins, color=colors)
    plt.xticks(bins)
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Number of Samples")
    plt.legend(["hits", "misses"])

    if output_file:
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        fig.savefig(output_file, bbox_inches="tight")


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
