import os
from contextlib import nullcontext
from typing import List, Union
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np

from leaderboard.nlu import collect_results

REPORTED_TIMES = [
    ("times-train_all", "training time (all components)"),
    ("times-train_all_classifiers", "training time (classifiers only)"),
]
# suffixes of measurements that are recorded per rasa component
# i.e. for "intent" or an entity extractor
REPORTED_PER_COMPONENT = [
    ("labels--", "number of classes (i.e. unique intents or entities)"),
    ("weighted avg-f1-score", "weighted avg. F1 score"),
    ("weighted avg-precision", "weighted avg. precision"),
    ("weighted avg-recall", "weighted avg. recall"),
    ("accuracy--", "accuracy"),
]
REPORTED_PER_COMPONENT += [
    (f"{avg_type} avg-{metric.lower().replace(' ','-')}", f"{avg_type} avg. {metric}")
    for avg_type in ["micro", "macro"]
    for metric in ["F1 score", "recall", "precision"]
]
# metrics reported per intent / entity (and extractor)
REPORTED_PER_CLASS = [
    "f1-score",
    "precision",
    "recall",
    "support",
    "confused_with",
]

COL_MODEL_NAME = "model-name"

PLOT_TYPES = ["scatter", "lines", "boxes"]

# mapping of column names for hyperparameters used in report files of
# experiments (with the 1st and 2nd level of column name concatenated by '-')
# to proper names to be used in plots
COL_HYPERPARAM_TO_NAME = {
    "param-train_exclusion_fraction": "Fraction of Excluded Training Data",
}


def sort_config_names(config_names: List[str]) -> List[str]:
    """Helper function to permute the given configuration names in a specific way.

    E.g., configuration names not containing the substrings "diet" and "bert" will
    be listed first, while those with
    Used to enforce a certain order of results in the legends.

    Args:
        config_names: list of some model
    """
    conditions = lambda name: [
        "diet" not in name and "bert" not in name,  # lightweight, non-diet first
        "diet" not in name,  # non-diet version next
        "with_transformer" not in name,  # of those with diet, the non-transformer first
        True,  # ... and the rest
    ]
    return sorted(config_names, key=lambda name: (conditions(name).index(True), name))


def _convert_to_label(component_prefix: str, col_suffix_y: str) -> str:
    if component_prefix == "intent":
        return f"Intent Classification / {col_suffix_y}"
    else:
        return f"Entity Extraction via {component_prefix} / {col_suffix_y}"


def generate_plots(
    in_file: str,
    col_x: str,
    label_x: str,
    save: bool,
    overwrite: bool,
    scatter: bool,
    lines: bool,
    boxes: bool,
    num_observations: bool,
) -> None:
    """Generate plots of `col_x` vs known `REPORTED_*` values.

    Args:
        in_file: report produced by the collect_results script
        col_x: name of column in `sub_df`
        label_x: name of column in `sub_df`
        label_x: name to be used in plots instead of `col_x`
        save: whether to just show or output the resulting plots; output filenames
          are generated programmatically (i.e. output files can be found in the same
          location as the input file and input_filename is used as prefix for output
          filenames)
        overwrite: set to True to allow output files to be overwritten without any
           warning
        scatter: set to True to produce scatter plots
        lines: set to True to produce line plots
        boxes: set to True to produce box plots
        num_observations: set to True to produce a plot showing the number of
          (non-nan) observations per unique `x`
    """

    # load data and flatten columns
    df_report = pd.read_csv(in_file, header=[0, 1])
    collect_results.add_total_train_times(df_report)
    df_report.columns = [f"{col[0]}-{col[1]}" for col in df_report.columns]

    # construct output path
    out_path, in_name = os.path.split(in_file)
    out_name_pattern = ".".join(in_name.split(".")[:-1]) + f"_{col_x}" + "_{}.pdf"
    out_file_pattern = os.path.join(out_path, out_name_pattern)

    def get_output_file(tag: str) -> str:
        output_file = out_file_pattern.format(tag)
        if not overwrite and os.path.exists(output_file):
            raise RuntimeError(f"File {output_file} already exists. Stop.")
        return output_file

    # static plot options
    common_kwargs = dict(
        scatter=scatter,
        lines=lines,
        boxes=boxes,
        num_observations=False,
        col_x=col_x,
        label_x=label_x,
    )

    out_file = get_output_file("train_times")
    with (nullcontext() if not save else PdfPages(out_file)) as context:

        for col_y, label_y in REPORTED_TIMES:
            _generate_plots(
                df=df_report,
                col_y=col_y,
                label_y=label_y,
                context=context,
                **common_kwargs,
            )

    # Rasa prepends the prefix "intent" to all reports that are about intents
    # (regardless of what classifier has been used), while entity extractor metrics
    # are prepended by the name of the entity extractor. Rasa does not produce any
    # aggregated results if there are multiple entity extractors.
    arbitrary_suffix = REPORTED_PER_COMPONENT[0][0]
    component_prefixes = sorted(
        set(
            col[: -len(arbitrary_suffix) - 1]  # remove leading underscore
            for col in df_report.columns
            if col.endswith(arbitrary_suffix)
        )
    )

    out_file = get_output_file("metrics")
    with (nullcontext() if not save else PdfPages(out_file)) as context:

        # Determine order of plots
        plot_order = []
        # ... first list all plots regarding intents ...
        if "intent" in component_prefixes:
            plot_order.extend(
                [
                    ("intent", col_label, common_kwargs)
                    for col_label in REPORTED_PER_COMPONENT
                ]
            )
        # ... then, per metric describing an entity and per chosen plot type, loop over
        # all extractors. This ensures the plots you can compare directly are closeby.
        for col_label in REPORTED_PER_COMPONENT:
            for plot_type in PLOT_TYPES:
                if not common_kwargs[plot_type]:
                    continue
                for component_prefix in component_prefixes:
                    if component_prefix == "intent":
                        continue
                    kwargs = common_kwargs.copy()
                    for arb_plot_type in PLOT_TYPES:
                        kwargs[arb_plot_type] = False
                    kwargs[plot_type] = True  # only this.
                    plot_order.append((component_prefix, col_label, kwargs))

        # Plot them!
        for component_prefix, (col_suffix_y, label_suffix_y), kwargs in plot_order:

            col_y = f"{component_prefix}_{col_suffix_y}"
            label_y = _convert_to_label(component_prefix, label_suffix_y)

            # special case: for the sanity check of the number of classes, we
            # just do a scatter plot - always
            if col_suffix_y == "labels--":
                kwargs_tweaked = kwargs.copy()
                kwargs_tweaked["scatter"] = True
                kwargs_tweaked["lines"] = False
                kwargs_tweaked["boxes"] = False
            else:
                kwargs_tweaked = kwargs

            _generate_plots(
                df=df_report,
                col_y=col_y,
                context=context,
                label_y=label_y,
                **kwargs_tweaked,
            )

    if num_observations:
        out_file = get_output_file("observation_numbers")
        with (nullcontext() if not save else PdfPages(out_file)) as context:

            for component_prefix in component_prefixes:
                for col_suffix_y, label_suffix_y in REPORTED_PER_COMPONENT:
                    col_y = f"{component_prefix}_{col_suffix_y}"
                    label_y = _convert_to_label(component_prefix, col_suffix_y)
                    _generate_plots(
                        df=df_report,
                        col_x=col_x,
                        label_x=label_x,
                        col_y=col_y,
                        label_y=label_y,
                        context=context,
                        num_observations=True,
                    )


def _show_or_save(context: Union[PdfPages, nullcontext]) -> None:
    plt.tight_layout()
    if isinstance(context, PdfPages):
        context.savefig(pad_inches=1)
    else:
        plt.show()
    plt.close()


def _generate_plots(
    df: pd.DataFrame,
    col_x: str,
    label_x: str,
    col_y: str,
    label_y: str,
    context: Union[PdfPages, nullcontext],
    scatter: bool = False,
    lines: bool = False,
    boxes: bool = False,
    num_observations: bool = False,
) -> None:
    """Generate various for `col_x` vs `col_y` grouped by `COL_MODEL_NAME`.

    Args:
        df: pandas dataframe containing columns `col_x`, `col_y`, and
            `COL_MODEL_NAME`
        col_x: name of column in `sub_df`
        col_y: name of column in `sub_df`
        label_x: name to be used in plots instead of `col_x`
        label_y: name to be used in plots instead of `col_y`
        context: either a null context (then plots are only shown) or PdfPages context
          (used for generating output files)
        scatter: set to True to produce scatter plot
        lines: set to True to produce line plot
        boxes: set to True to produce box plot
        num_observations: set to True to produce a plot showing the number of
          (non-nan) observations per unique `x`
    """
    sub_df = df[[COL_MODEL_NAME, col_x, col_y]].copy()
    sorted_models = sort_config_names(sub_df[COL_MODEL_NAME].unique())

    # Check how
    nans = sub_df.isna().max(axis=0).sum()
    sub_df = sub_df.dropna()

    # sanity check: how many observations are there per model
    if num_observations:
        num_y_per_x_and_model = sub_df[[COL_MODEL_NAME, col_x]].value_counts()
        pd.DataFrame(num_y_per_x_and_model).plot.bar(
            title=f"number of observations ({col_y})"
        )
        _show_or_save(context=context)

    title = (
        f"{label_y} / varying sizes of training data\n"
        f"(showing {len(sub_df)} observations in total / found {nans} NaNs)"
    )

    if scatter:
        plt.figure(figsize=(10, 5))
        ax = sns.catplot(
            data=sub_df,
            orient="h",
            x=col_y,
            y=col_x,
            hue=COL_MODEL_NAME,
            hue_order=sorted_models,
            height=5,
            aspect=2,
            legend=False,
        )
        ax.set(xlabel=label_y, ylabel=label_x, title=title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        _show_or_save(context=context)

    if lines:
        plt.figure(figsize=(10, 5))
        ax = sns.lineplot(
            data=sub_df,
            x=col_x,
            y=col_y,
            hue=COL_MODEL_NAME,
            hue_order=sorted_models,
            marker="o",
            ci="sd",
            alpha=0.6,
        )
        ax.set(xlabel=label_x, ylabel=label_y, title=title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        _show_or_save(context=context)

    if boxes:
        plt.figure(figsize=(10, 5))
        ax = sns.boxplot(
            data=sub_df,
            orient="h",
            x=col_y,
            y=col_x,
            hue=COL_MODEL_NAME,
            hue_order=sorted_models,
        )
        ax.set(xlabel=label_y, ylabel=label_x, title=title)
        # fig.set_xscale("log")
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        _show_or_save(context=context)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot collected results.")
    parser.add_argument("filepath", type=str, help="path to input file")
    parser.add_argument("x", type=str, help="hyperparameter")
    parser.add_argument("--save", type=bool, help="save plots", default=True)
    parser.add_argument("--scatter", type=bool, help="add scatter plot", default=True)
    parser.add_argument("--lines", type=bool, help="add line plot", default=True)
    parser.add_argument("--boxes", type=bool, help="add box plots", default=False)
    parser.add_argument(
        "--overwrite",
        type=bool,
        help="allow overwriting files if " "saving",
        default=False,
    )
    parser.add_argument(
        "--num_observations",
        type=bool,
        help="add plot on the number of observations (separate file)",
        default=True,
    )
    args = parser.parse_args()

    label_for_x = COL_HYPERPARAM_TO_NAME.get(args.x, None)
    if label_for_x is None:
        raise ValueError(
            f"Unknown hyperparameter column {args.x}. Available options "
            f"are {sorted(COL_HYPERPARAM_TO_NAME.keys())}. "
            f"Please add a name for the column to the `COL_HYPERPARAM_TO_NAME` "
            f"mapping, if it is a new key. Otherwise, correct the typo :)."
        )

    generate_plots(
        in_file=args.filepath,
        save=args.save,
        label_x=label_for_x,
        col_x=args.x,
        scatter=args.scatter,
        boxes=args.boxes,
        lines=args.lines,
        overwrite=args.overwrite,
        num_observations=args.num_observations,
    )
