:desc: Learn how to integrate open source chatbot platform Rasa into a
       Jupyter notebooks, alongside all your machine learning code. a
 a
.. _jupyter-notebooks: a
 a
Jupyter Notebooks a
================= a
 a
.. edit-link:: a
 a
This page contains the most important methods for using Rasa in a Jupyter notebook. a
 a
Running asynchronous Rasa code in Jupyter Notebooks requires an extra requirement, a
since Jupyter Notebooks already run on event loops. Install this requirement in a
the command line before launching jupyter: a
 a
.. code-block:: bash a
 a
   pip install nest_asyncio a
 a
Then in the first cell of your notebook, include: a
 a
.. runnable:: a
 a
   import nest_asyncio a
 a
   nest_asyncio.apply() a
   print("Event loop ready.") a
 a
 a
First, you need to create a project if you don't already have one. a
To do this, run this cell, which will create the ``test-project`` directory and make it a
your working directory: a
 a
.. runnable:: a
 a
   from rasa.cli.scaffold import create_initial_project a
   import os a
 a
   project = "test-project" a
   create_initial_project(project) a
 a
   # move into project directory and show files a
   os.chdir(project) a
   print(os.listdir(".")) a
 a
 a
To train a model, you will have to tell the ``train`` function a
where to find the relevant files. a
To define variables that contain these paths, run: a
 a
 a
.. runnable:: a
 a
   config = "config.yml" a
   training_files = "data/" a
   domain = "domain.yml" a
   output = "models/" a
   print(config, training_files, domain, output) a
 a
 a
 a
 a
Train a Model a
~~~~~~~~~~~~~ a
 a
Now we can train a model by passing in the paths to the ``rasa.train`` function. a
Note that the training files are passed as a list. a
When training has finished, ``rasa.train`` returns the path where the trained model has been saved. a
 a
 a
 a
.. runnable:: a
 a
   import rasa a
 a
   model_path = rasa.train(domain, config, [training_files], output) a
   print(model_path) a
 a
 a
 a
 a
Chat with your assistant a
~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
To start chatting to an assistant, call the ``chat`` function, passing a
in the path to your saved model: a
 a
 a
.. runnable:: a
 a
   from rasa.jupyter import chat a
   chat(model_path) a
 a
 a
 a
Evaluate your model against test data a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ a
 a
Rasa has a convenience function for getting your training data. a
Rasa's ``get_core_nlu_directories`` is a function which a
recursively finds all the stories and NLU data files in a directory a
and copies them into two temporary directories. a
The return values are the paths to these newly created directories. a
 a
.. runnable:: a
 a
   import rasa.data as data a
   stories_directory, nlu_data_directory = data.get_core_nlu_directories(training_files) a
   print(stories_directory, nlu_data_directory) a
 a
 a
 a
To test your model, call the ``test`` function, passing in the path a
to your saved model and directories containing the stories and nlu data a
to evaluate on. a
 a
.. runnable:: a
 a
   rasa.test(model_path, stories_directory, nlu_data_directory) a
   print("Done testing.") a
 a
 a
The results of the core evaluation will be written to a file called ``results``. a
NLU errors will be reported to ``errors.json``. a
Together, they contain information about the accuracy of your model's a
predictions and other metrics. a
 a
.. runnable:: a
 a
   if os.path.isfile("errors.json"): a
       print("NLU Errors:") a
       print(open("errors.json").read()) a
   else: a
       print("No NLU errors.") a
 a
   if os.path.isdir("results"): a
         print("\n") a
         print("Core Errors:") a
         print(open("results/failed_stories.md").read()) a
 a
.. juniper:: a
  :language: python a
 a