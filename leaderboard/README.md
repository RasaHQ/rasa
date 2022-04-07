# Leaderboard 

## Prerequisites

In addition to Rasa, we'll use [Hydra](hydra.cc) and pandas for configuration 
management and 
evaluations. Both can be installed via: 
```
pip install -r requirements.txt
```

## Getting Started

1. Take a look at our [demo notebook](./demo/Run-Multiple-Intent-Experiments.ipynb)
   on how to kick-off and inspect multiple intent experiments.
2. Get to know the code:
   1. [utils/experiment.py](./utils/experiment.py) shows the basic structure of 
      every experiment and contains the boilerplate code for hydra.
   2. [nlu/exp_0_stratify.py](./nlu/exp_0_stratify_intents.py) is one concrete 
      example of how an experiment is defined.
   3. [nlu/README.md](nlu/README.md) gives an overview on the different ways in 
      which the experiment scripts can be used.

## Structure

| folder  | contains                                   |
|---------|--------------------------------------------|
| demo    | snippets and notebooks for getting started |
| utils   | shared utility functions                   |
| nlu     | definitions of nlu experiments             |
| workspace | scripts to run concrete experiments + EDA  |