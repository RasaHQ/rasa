.. _section_configuration:

Configuration
==================================

You can provide options to rasa NLU through:

- a json-formatted config file
- environment variables
- command line arguments

Environment variables override options in your config file, 
and command line args will override any options specified elsewhere.
Environment variables are capitalised and prefixed with ``RASA_``, 
so the option ``backend`` is specified with the ``RASA_BACKEND`` env var.

Here is a list of all rasa NLU configuration options:

- ``backend`` : (optional) if present, can be ``mitie`` or ``spacy_sklearn``
- ``config`` : configuration file (can only be set as env var or command line option)
- ``data`` : file containing training data.
- ``language`` : language of your app, can be ``en`` (English) or ``de`` (German).
- ``mitie_file`` : file containing ``total_word_feature_extractor.dat`` (see :ref:`backends`)
- ``path`` : where trained models will be saved.
- ``port`` : port on which to run server.
- ``server_model_dir`` : dir containing the model to be used by server.
- ``write`` : file where logs will be saved


You can also persist your trained models to S3, and fetch them from there. This requires the following options to be set.
Note these do not have a ``RASA_`` prefix when set as environment vars.

- ``aws_region`` : region for S3 bucket where models are saved
- ``bucket_name`` : name of S3 bucket where models are saved
- ``aws_secret_access_key`` : secret
- ``aws_access_key_id`` : key
