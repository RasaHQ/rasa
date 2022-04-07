# NLU Experiments

## Preliminaries

1. This folder contains **definitions of generic experiments**. 
2. Scripts defining a concrete run of the experiments (i.e. specific configurations on 
specific datasets) are located in [../workspace/runs](../workspace/runs).

## Reminder for Reproducible Results

1. To create a new experiment **do not alter existing experiment scripts**
2. Re-use parts of old experiments via subclassing.
3. Have a look at 1. again.

## Running Experiments

### 1. Multi-Runs

For how to easily define and run multiple experiments, have a look at the demo 
notebook 
[Run-Multiple-Intent-Experiments.ipynb](../demo/Run-Multiple-Intent-Experiments.ipynb).

### 2. Via Command Line

Have a look at
- the [Hydra tutorials](https://hydra.cc/docs/next/tutorials/intro/)
- the boilerplate code we define in [utils/experiment.py](../utils/experiment.py)
