# Jupyter Notebooks

This page contains the most important methods for using Rasa in a Jupyter notebook.

Running asynchronous Rasa code in Jupyter Notebooks requires an extra requirement,
since Jupyter Notebooks already run on event loops. Install this requirement in
the command line before launching jupyter:

```
pip install nest_asyncio
```

Then in the first cell of your notebook, include:

First, you need to create a project if you don’t already have one.
To do this, run this cell, which will create the `test-project` directory and make it
your working directory:

To train a model, you will have to tell the `train` function
where to find the relevant files.
To define variables that contain these paths, run:

## Train a Model

Now we can train a model by passing in the paths to the `rasa.train` function.
Note that the training files are passed as a list.
When training has finished, `rasa.train` returns the path where the trained model has been saved.

## Chat with your assistant

To start chatting to an assistant, call the `chat` function, passing
in the path to your saved model:

## Evaluate your model against test data

Rasa has a convenience function for getting your training data.
Rasa’s `get_core_nlu_directories` is a function which
recursively finds all the stories and NLU data files in a directory
and copies them into two temporary directories.
The return values are the paths to these newly created directories.

To test your model, call the `test` function, passing in the path
to your saved model and directories containing the stories and nlu data
to evaluate on.

The results of the core evaluation will be written to a file called `results`.
NLU errors will be reported to `errors.json`.
Together, they contain information about the accuracy of your model’s
predictions and other metrics.
