# Leaderboard 

## Guidelines for Reproducible Results

1. To create a new experiment **do not alter existing experiment scripts**
2. Re-use parts of old experiments via subclassing.
3. Have a look at 1. again.

## Prerequisites

```sh 
    pip install hydra-core --upgrade
    pip install omegaconf
    #pip install hydra-optuna-sweeper --upgrade
```

## Getting Started

### A Quick Intro to Hydra

Configurations are managed via Hydra. It is recommended to have a look at the basic 
Hydra tutorials at https://hydra.cc/docs/next/tutorials/intro/ before you start.

To give you a glimpse at what Hydra is helpful for.  

If you just run the experiment script via
```sh
  python bla_experiment.py
```
then the results for this single experiment run will end up in
```sh
    <cwd>/outputs/<current date>/<time when experiment was started>
```
and the default configs will be used (if there are no missing values).

Nothing exciting so far. But...

With the *same* experiment script, we can generate leaderboard entries for multiple 
configurations and store the results in sub-folders whose names reflect the respective 
configuration, by simply running
```sh
    python run.py --multirun nlu.save_plots=True,False \
       hydra.sweep.subdir='${hydra.job.override_dirname}'
       hydra.sweep.dir=./custom-folder
```
This will result in the following output structure:
```sh
    <path to custom-folder>/
        nlu.save_plots=False/
            .hydra/ # hydra config
            run.log
            # whatever else we store here
        nlu.save_plots=True/
            ...
```
For more options, see https://hydra.cc/docs/configure_hydra/workdir/.

And if this doesn't spark enough joy yet. Take a look at the "sweepers" that Hydra 
offers and how you can add hyperparameter search with minimal code changes.
More information can be found, for example, at https://hydra.cc/docs/plugins/nevergrad_sweeper/.

Now that you're really excited, plase note: Rasa does not use hydra. So we can't 
hyperparameter optimization for Rasa Components will not work out of the box. To 
implement this, you'll need to create a configuration with the desired hyperparameters 
and a custom experiment script that passes on those hyperparameters to Rasa.

### Executing Experiments

#### Running a Single Experiment

Test the "base" intent experiment, by running 

```sh
cd <path to rasa>/leaderboard/nlu/base
python intent_experiment.py model.config_path=../../examples/rules/config2.yml model.name="rule_config" data.data_path=../../examples/rules/data/nlu.yml data.domain_path=../../examples/rules/domain.yml experiment.drop_intents_with_less_than=0
```

And checkout the results created in `<path to rasa>/leaderboard/nlu/base/outputs`.
In addition to what is described in the quick Hydra info, this folder should contain:
```sh
    outputs/
      <current date>/
        <current time>/
          .hydra/
            config.yaml # contains the experiment configuration
            hydra.yaml # full hydra configuration
            overrides.yaml
          data/
            train.yml 
            test.yml 
          model/
            <model name that was set in command line>.tar.gz
          report/ # contains results of rasa test
          <file with same name as experiment script>.yml # containing logs  
```


#### Sweeping over Multiple Configurations / Datasets

- Option 1: Via Command Line 

    For an example for how to vary parameters via the commandline, take a look at 
    the `multirun` example mentioned in 
    [Section 1: A Quick Intro To Hydra](#1-a-quick-intro-to-hydra).

- Option 2: Via a Config File and a Hydra Sweeper

    TBD: sweeper config file

- Option 3: Workaround

    To benefit from the features of `hydra.main` and avoid configuring everything in the 
commandline, simply invoke the experiment script via python.

    For more information, see `leaderboard.nlu.base.intent_experiment.multirun` and 
  the corresponding demo in 
  `leaderboard/demo/example1/Run-Multiple-IntentExperiments.ipynb`.

#### Note

- Have a look at tha alternative ways of configuring Hydra via config files instead 
  of dataclasses to get sweepers to work more easily. 
- If certain combinations of hyperparameters are invalid, then you have two options:
  1. design the experiment such that it fails in that case
  2. design the configuration such that
  3. do not forget to not mess with old experiment scripts to not break reproducibility 


### Collecting and Aggregating Results (TBD)