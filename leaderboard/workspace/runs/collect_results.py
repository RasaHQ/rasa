import glob
import os
from leaderboard.nlu import evaluation

results_path = "/home/jupyter/results"

if __name__ == "__main__":

    for experiment in glob.glob(os.path.join(results_path, '*')):
        df_full = evaluation.results2df(os.path.join(results_path, experiment))
        filename = os.path.basename(f"./{experiment}.csv")
        df_full.to_csv(filename)