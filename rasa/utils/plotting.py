import logging
import itertools
import os

import numpy as np
from typing import Dict, List, Set, Text, Optional, Union, Any

from matplotlib.axes import Axes
import matplotlib
from matplotlib.ticker import FormatStrFormatter

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
    """Plot a side-by-side comparative histogram of the confidence distribution (misses and hits).

    Args:
        hist_data: histogram data
        title: title of the plot
        output_file: output file to save the plot to
    """
    import matplotlib.pyplot as plt

    plt.gcf().clear()

    # Wine-ish colour for the confidences of hits.
    # Blue-ish colour for the confidences of misses.
    colors = ["#009292", "#920000"]
    n_bins = 25
    max_value = max(
        [max(hist_data[0], default=0), max(hist_data[1], default=0)], default=0
    )
    min_value = min(
        [min(hist_data[0], default=0), min(hist_data[1], default=0)], default=0
    )

    bin_width = (max_value - min_value) / n_bins
    bins = [min_value + (i * bin_width) for i in range(1, n_bins + 1)]

    binned_data_sets = [np.histogram(d, bins=bins)[0] for d in hist_data]

    max_xlims = [max(binned_data_set) for binned_data_set in binned_data_sets]
    max_xlims = [xlim + np.ceil(0.25 * xlim) for xlim in max_xlims]  # padding

    min_ylim = (
        bins[
            min(
                [
                    (binned_data_set != 0).argmax(axis=0)
                    for binned_data_set in binned_data_sets
                ]
            )
        ]
        - bin_width
    )

    max_ylim = max(bins) + bin_width

    yticks = [float("{:.2f}".format(x)) for x in bins]

    centers = 0.5 * (0.05 + (bins + np.roll(bins, 0))[:-1])
    heights = 0.75 * np.diff(bins)

    fig, axes = plt.subplots(ncols=2, sharey=True)
    axes[0].barh(
        centers,
        binned_data_sets[0],
        height=heights,
        align="center",
        color=colors[0],
        label="hits",
    )
    axes[0].set(title="Correct")
    axes[1].barh(
        centers,
        binned_data_sets[1],
        height=heights,
        align="center",
        color=colors[1],
        label="misses",
    )

    axes[1].set(title="Wrong")

    axes[0].set(yticks=yticks, xlim=(0, max_xlims[0]), ylim=(min_ylim, max_ylim))
    axes[1].set(yticks=yticks, xlim=(0, max_xlims[1]), ylim=(min_ylim, max_ylim))

    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].yaxis.set_minor_formatter(FormatStrFormatter("%.2f"))

    axes[0].invert_xaxis()
    axes[0].yaxis.tick_right()

    fig.subplots_adjust(
        wspace=0.17
    )  # get the graphs exactly far enough apart for yaxis labels
    fig.suptitle(title, fontsize="x-large", fontweight="bold")

    # Add hidden plot to correctly add x and y labels
    fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.ylabel("Confidence")
    plt.xlabel("Number of Samples")

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
    metric: Text,
    output_file: Text,
) -> None:
    """Plot the gain/loss curve per intent.

    Args:
        augmentation_summary: Performance summary dictionary.
        metric: Metric to plot, must be one of "precision", "recall", or "f1-score".
        output_file: Output file for plot.
    """
    import matplotlib.pyplot as plt

    accuracy = augmentation_summary.get("accuracy", None)

    totals_keys = {"weighted avg", "micro avg", "macro avg", "accuracy"}
    intents = list(augmentation_summary.keys())

    metric_key = f"{metric}_change"
    performance = np.array(
        [d[metric_key] for d in augmentation_summary.values() if metric_key in d]
    )

    if accuracy is not None:
        acc_idx = intents.index("accuracy")
        performance = np.insert(performance, acc_idx, accuracy["accuracy_change"])

    num_intents = len(intents)
    ind = np.arange(num_intents)

    # Try to autoscale the figure size to leave enough room for all intents
    fig_size = (10, 10)
    if 30 <= num_intents < 60:
        fig_size = (12, 20)
    elif num_intents >= 60:
        fig_size = (15, 30)

    plt.figure(figsize=fig_size)
    plt.xlabel(f"Performance Change ({metric})", fontsize=16)
    plt.ylabel("Intent", fontsize=16)
    plt.title("NLU Data Augmentation Summary")
    perf_bar = plt.barh(ind, performance)

    for idx in range(num_intents):
        if performance[idx] < 0.0:
            perf_bar[idx].set_color("#920000")
        else:
            perf_bar[idx].set_color("#009292")
        if intents[idx] in totals_keys:
            perf_bar[idx].set_hatch("*")
            # The colour of the hatch is determined by the edge colour property, so in order to make the hatch visible,
            # we need ot set the edge colour explicitly
            perf_bar[idx].set_edgecolor("white")

    _autolabel(perf_bar)
    plt.ylim((-1, num_intents))
    plt.xlim((np.min(performance) - 0.2, np.max(performance) + 0.2))
    plt.yticks(ind, intents, fontsize=14)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")


def _autolabel(rects: Axes.bar) -> None:
    """Attach a text label above each bar in *rects*, displaying its height.

    Args:
        rects: The barplot object.
    """
    import matplotlib.pyplot as plt

    for rect in rects:
        width = rect.get_width()
        offset = 0.0
        if width > 0.0:
            offset = 0.07
        elif width < 0.0:
            offset = -0.075

        plt.annotate(
            f"{width:.2f}",
            xy=(width + offset, rect.get_y() + (rect.get_height() / 5)),
            xytext=(0, 1),  # horizontal offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )
