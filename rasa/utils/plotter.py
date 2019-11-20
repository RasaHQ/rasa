from pathlib import Path
from typing import Union, List, Text

import numpy as np
import csv


# to enable %matplotlib inline if running in ipynb
from IPython import get_ipython

ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic("matplotlib", "inline")


import matplotlib.pyplot as plt


class Plotter(object):
    """
    Plots training parameters (loss, f-score, and accuracy) and training weights over time.
    Input files are the output files 'loss.tsv' and 'weights.txt' from training either a sequence tagger or text
    classification model.
    """

    @staticmethod
    def _extract_evaluation_data(
        file_name: Text, score: str = "loss", prefix: str = "i"
    ) -> dict:
        training_curves = {"train": [], "val": []}

        with open(file_name, "r") as tsvin:
            tsvin = csv.reader(tsvin, delimiter="\t")

            # determine the column index of loss, f-score and accuracy for train, dev and test split
            row = next(tsvin, None)

            score = score.upper()

            TRAIN_SCORE = (
                row.index(f"{prefix.upper()}_{score.upper()}")
                if f"{prefix.upper()}_{score.upper()}" in row
                else None
            )
            VAL_SCORE = (
                row.index(f"VAL_{prefix.upper()}_{score.upper()}")
                if f"VAL_{prefix.upper()}_{score.upper()}" in row
                else None
            )

            # then get all relevant values from the tsv
            for row in tsvin:

                if TRAIN_SCORE is not None:
                    if row[TRAIN_SCORE] != "_":
                        training_curves["train"].append(float(row[TRAIN_SCORE]))

                if VAL_SCORE is not None:
                    if VAL_SCORE < len(row) and row[VAL_SCORE] != "_":
                        training_curves["val"].append(float(row[VAL_SCORE]))
                    else:
                        training_curves["val"].append(0.0)

        return training_curves

    def plot_training_curves(self, file_name: Union[Text], output_folder: Text):
        if type(output_folder) is str:
            output_folder = Path(output_folder)

        metrics = {
            "intent": {"scores": ["loss", "acc"], "prefix": "i"},
            "entity": {"scores": ["loss", "f1"], "prefix": "e"},
        }

        for metric_name, metric_values in metrics.items():

            fig = plt.figure(figsize=(15, 10))

            prefix = metric_values["prefix"]
            scores = metric_values["scores"]

            output_path = output_folder / f"training_{metric_name}.png"

            for i, score in enumerate(scores):
                training_curves = self._extract_evaluation_data(
                    file_name, score, prefix
                )

                plt.subplot(len(scores), 1, i + 1)
                if training_curves["train"]:
                    x = np.arange(0, len(training_curves["train"]))
                    plt.plot(
                        x,
                        training_curves["train"],
                        label=f"train {metric_name} {score}",
                    )
                if training_curves["val"]:
                    x = np.arange(0, len(training_curves["val"]))
                    plt.plot(
                        x, training_curves["val"], label=f"val {metric_name} {score}"
                    )

                plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
                plt.ylabel(f"{metric_name} {score}")
                plt.xlabel("epochs")

            # save plots
            plt.tight_layout(pad=1.0)
            plt.savefig(output_path, dpi=300)
            print(
                f"Loss and acc plots are saved in {output_path}"
            )  # to let user know the path of the save plots
            plt.close(fig)
