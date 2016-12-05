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

- ``backend`` :  if present, can be ``mitie`` or ``spacy_sklearn``
- ``config`` : configuration file (can only be set as env var or command line option)
- ``data`` : file containing training data.
- ``emulate`` :  service to emulate. can be ``wit``, ``luis``, or ``api``.
- ``language`` : language of your app, can be ``en`` (English) or ``de`` (German).
- ``mitie_file`` : file containing ``total_word_feature_extractor.dat`` (see :ref:`section_backends`)
- ``path`` : where trained models will be saved.
- ``port`` : port on which to run server.
- ``server_model_dir`` : dir containing the model to be used by server.
- ``token`` :  if set, all requests to server must have a ``?token=<token>`` query param. see :ref:`section_auth`
- ``write`` : file where logs will be saved


If you want to persist your trained models to S3, there are additional configuration options,
see :ref:`section_persistence`