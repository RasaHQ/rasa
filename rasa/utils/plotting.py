import logging
import itertools
import os

import numpy as np
from typing import Dict, List, Set, Text, Optional, Union, Any

from matplotlib.axes import Axes
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


def plot_intent_augmentation_summary(
    augmentation_summary: Dict[Text, Dict[Text, float]],
    changed_intents: Set[Text],
    metric: Text,
    output_file: Text,
) -> None:
    """Plot the gain/loss curve per intent.

    Args:
        augmentation_summary: Performance summary dictionary.
        changed_intents: Intents that have not been augmented, but where performance changed.
        metric: Metric to plot, must be one of "precision", "recall", or "f1-score".
        output_file: Output file for plot.
    """
    import matplotlib.pyplot as plt

    totals_keys = {"weighted avg", "micro avg", "macro avg"}
    intents = list(augmentation_summary.keys())
    num_intents = len(intents)

    ind = np.arange(num_intents)
    performance = np.array(
        list(map(lambda d: d[f"{metric}_change"], augmentation_summary.values()))
    )

    plt.figure(figsize=(10, 10))
    plt.xlabel(f"Performance Change ({metric})", fontsize=16)
    plt.ylabel("Intent", fontsize=16)
    perf_bar = plt.barh(ind, performance)

    for idx in range(num_intents):
        if performance[idx] < 0.0:
            perf_bar[idx].set_color("lightcoral")
        else:
            perf_bar[idx].set_color("lightgreen")
        if intents[idx] in totals_keys:
            perf_bar[idx].set_hatch("*")
            # The colour of the hatch is determined by the edge colour property, so in order to make the hatch visible,
            # we need ot set the edge colour explicitly
            perf_bar[idx].set_edgecolor("black")

    _autolabel(perf_bar)
    plt.ylim((-1, num_intents))
    plt.xlim((np.min(performance) - 0.2, np.max(performance) + 0.2))
    plt.yticks(ind, intents, fontsize=14)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.savefig(output_file, bbox_inches="tight")


def _autolabel(rects: Axes.bar) -> None:
    """Attach a text label above each bar in *rects*, displaying its height."""
    import matplotlib.pyplot as plt

    for rect in rects:
        width = rect.get_width()
        offset = 0.0
        if width > 0.0:
            offset = 0.045
        elif width < 0.0:
            offset = -0.045

        plt.annotate(
            f"{width:.2f}",
            xy=(width + offset, rect.get_y() + (rect.get_height() / 5)),
            xytext=(0, 1),  # horizontal offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )
