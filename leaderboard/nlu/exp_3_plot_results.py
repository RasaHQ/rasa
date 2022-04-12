import os
from contextlib import nullcontext
from typing import List
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import numpy as np

from leaderboard.nlu import collect_results

MEASUREMENTS = [
    ("times-train_all", "training time (all components)"),
    ("times-train_all_classifiers", "training time (classifiers only)"),
    ("weighted avg-f1-score", "weighted avg. F1 score"),
    ("weighted avg-precision", "weighted avg. precision"),
    ("weighted avg-recall", "weighted avg. recall"),
]
MEASUREMENTS += [
    (f"{avg_type} avg-{metric.lower().replace(' ','-')}", f"{avg_type} avg. {metric}")
    for avg_type in ["micro", "macro"]
    for metric in ["F1 score", "recall", "precision"]
]
LABEL_X = "Fraction of Excluded Training Data"
COL_X = "param-train_exclusion_fraction"
COL_MODEL_NAME = "model-name"


def sort_model_names(model_names: List[str]) -> List[str]:
    models_unsorted = sorted(model_names)
    non_diet = [model for model in models_unsorted if "diet" not in model]
    diet_wo_transformer = [
        model for model in models_unsorted if "diet" in model and "without" in model
    ]
    models = non_diet + diet_wo_transformer
    models += [model for model in models_unsorted if model not in models]
    return models


def generate_plots(
    in_file: str, save: bool, scatter: bool, lines: bool, boxes: bool
) -> None:

    # load data
    results = pd.read_csv(in_file, header=[0, 1])
    collect_results.add_total_train_times(results)
    results.columns = [f"{col[0]}-{col[1]}" for col in results.columns]

    # sort labels in all plots as follows
    sorted_models = sort_model_names(results[COL_MODEL_NAME].unique())

    # output path
    out_path, in_name = os.path.split(in_file)
    out_name = ".".join(in_name.split(".")[:-1]) + ".pdf"
    out_file = os.path.join(out_path, out_name)

    with (nullcontext() if not save else PdfPages(out_file)) as pdf:

        def show_or_save():
            if save:
                pdf.savefig(pad_inches=1)
                plt.close()
            else:
                plt.show()

        for y_col, y_label in MEASUREMENTS:

            sub_df = results[[COL_MODEL_NAME, COL_X, y_col]].copy()
            nans = sub_df.isna().max(axis=0).sum()
            sub_df = sub_df.dropna()

            title = (
                f"{LABEL_X} for varying sizes of training data\n"
                f"({len(sub_df)} rows, exluding {nans} rows with NaNs)"
            )

            # sanity check: how many observations are there per
            num_y_per_x_and_model = sub_df[[COL_MODEL_NAME, COL_X]].value_counts()
            pd.DataFrame(num_y_per_x_and_model).plot.bar(title="number of measurements")
            plt.tight_layout()
            show_or_save()

            if scatter:
                plt.figure(figsize=(10, 5))
                ax = sns.catplot(
                    data=sub_df,
                    orient="h",
                    x=y_col,
                    y=COL_X,
                    hue=COL_MODEL_NAME,
                    hue_order=sorted_models,
                    height=5,
                    aspect=2,
                    legend=False,
                )
                ax.set(xlabel=y_label, ylabel=LABEL_X, title=title)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                plt.tight_layout()
                show_or_save()

            if lines:
                plt.figure(figsize=(10, 5))
                ax = sns.lineplot(
                    data=sub_df,
                    x=COL_X,
                    y=y_col,
                    hue=COL_MODEL_NAME,
                    hue_order=sorted_models,
                    marker="o",
                    ci="sd",
                    alpha=0.6,
                )
                ax.set(xlabel=LABEL_X, ylabel=y_label, title=title)
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                plt.tight_layout()
                show_or_save()

            if boxes:
                plt.figure(figsize=(10, 5))
                ax = sns.boxplot(
                    data=sub_df,
                    orient="h",
                    x=y_col,
                    y=COL_X,
                    hue=COL_MODEL_NAME,
                    hue_order=sorted_models,
                )
                ax.set(xlabel=y_label, ylabel=LABEL_X, title=title)
                # fig.set_xscale("log")
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
                plt.tight_layout()
                show_or_save()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot collected results.")
    parser.add_argument("filepath", type=str, help="path to input file")
    parser.add_argument("--save", type=bool, help="save plots", default=True)
    parser.add_argument("--scatter", type=bool, help="save plots", default=True)
    parser.add_argument("--lines", type=bool, help="save plots", default=True)
    parser.add_argument("--boxes", type=bool, help="save plots", default=True)
    args = parser.parse_args()

    generate_plots(
        in_file=args.filepath,
        save=args.save,
        scatter=args.scatter,
        boxes=args.boxes,
        lines=args.lines,
    )
