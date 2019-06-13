.. _jupyter-notebooks:

Jupyter Notebooks
=================

This page contains the most important methods for using Rasa in a Jupyter notebook.

You need to create a project if you don't already have one.
To do this, run:

.. runnable::
   import os
   from rasa.cli.scaffold import _create_initial_project

   project = "test-project"
   _create_initial_project(project)
   os.chdir(project)

Now that you have a project, the relevant files and folders exist.
To check this, run:

.. runnable::
   os.listdir(project)


To train a model, you will have to tell the ``train`` function
where to find the relevant files.
To define variables that contain these paths, run:

.. runnable::

   config = "config.yml"
   training_files = "data/"
   domain = "domain.yml"
   output = "models/"


Train a Model
~~~~~~~~~~~~~

Now we can train a model by passing in the paths to the ``rasa.train`` function.
Note that the training files are passed as a list.
When training has finished, ``rasa.train`` returns the path where the trained model has been saved.

.. runnable::

   import rasa

   model_path = rasa.train(domain, config, [training_files], output)


Chat with your assistant
~~~~~~~~~~~~~~~~~~~~~~~~

To start chatting to an assistant, call the ``chat`` function, passing
in the path to your saved model:

.. runnable::

   from rasa.jupyter import chat
   chat(model_path)


Evaluate your model against test data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rasa has a convenience function for getting your training data.
Rasa's ``get_core_nlu_directories`` is a convenience function which
recursively finds all the stories and nlu data files in a directory,
and copies them into two directories.
The return values are the paths to these newly created directories.

.. runnable::

   import rasa.data as data
   stories_directory, nlu_data_directory = data.get_core_nlu_directories(training_files)

To test your model, call the ``test`` function, passing in the path
to your saved model and directories containing the stories and nlu data
to evaluate on.

.. runnable::

   rasa.test(model_path, stories_directory, nlu_data_directory)

The results of the evaluation will be written to a file called ``results``.
This contains information about the accuracy of your model and other metrics.

.. runnable::

   ls results
